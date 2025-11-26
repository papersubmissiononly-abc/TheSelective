import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import gc
import logging

from models.common import compose_context, ShiftedSoftplus
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils.vina_rules import calc_vina

def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def normalize_vina_score(vina_score, min_score=-16.0, max_score=0.0):

    clamped = torch.clamp(vina_score, min_score, max_score)
    # Normalize to [0,1] and invert so that lower (better) scores become higher values
    normalized = 1.0 - (clamped - min_score) / (max_score - min_score)
    return normalized

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein', center_ligand=False):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        if center_ligand:
            # Center on ligand center of mass
            offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        else:
            # Center on protein center of mass (original behavior)
            offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return gumbel_noise + logits


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


########### Model ##############
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        
        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.loss_exp_weight = config.loss_exp_weight
        ## 새로 추가
        self.lambda_on = config.get('lambda_on', 1.0)
        self.lambda_off = config.get('lambda_off', 1.0)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        
        self.use_classifier_guide = config.use_classifier_guide

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # classifier guidance weight
        self.pos_classifier_grad_weight = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']
        self.center_ligand = getattr(config, 'center_ligand', False)  # True: center on ligand, False: center on protein

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        # Choose expert_pred architecture based on config
        use_dual_head = getattr(config, 'use_dual_head_ba', False)

        if use_dual_head:
            # Dual-head architecture: separate heads for on-target and off-target BA prediction
            self.expert_input_dim = self.hidden_dim * 2

            # Head 1: On-target BA prediction (protein-ligand interaction present)
            self.expert_pred_on = nn.Sequential(
                nn.Linear(self.expert_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            # Head 2: Off-target BA prediction (protein-ligand interaction absent)
            self.expert_pred_off = nn.Sequential(
                nn.Linear(self.expert_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )

            # Keep original expert_pred for backward compatibility
            self.expert_pred = self.expert_pred_on
            print(f"[INFO] Using DUAL-HEAD BA prediction architecture: input_dim={self.hidden_dim * 2}")

        elif getattr(config, 'use_original_expert', False):
            # Original architecture: expert_pred takes only hidden_dim input
            self.expert_input_dim = self.hidden_dim
            self.expert_pred = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            print(f"[INFO] Using ORIGINAL expert_pred architecture: input_dim={self.hidden_dim}")
        else:
            # Current architecture: expert_pred takes hidden_dim*2 input (ligand + protein)
            self.expert_input_dim = self.hidden_dim * 2
            self.expert_pred = nn.Sequential(
                nn.Linear(self.expert_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()  # 0~1 범위로 normalize (vina score style)
            )
            print(f"[INFO] Using CURRENT expert_pred architecture: input_dim={self.hidden_dim * 2}")

        self.use_dual_head = use_dual_head
        self.use_dual_head_sam_pl = getattr(config, 'use_dual_head_sam_pl', False)

        # Sam-PL dual head: original KGDiff vs non-interaction based prediction
        if self.use_dual_head_sam_pl:
            # Check which variant of dual-head architecture to use
            self.use_atom_level_cross_attn = getattr(config, 'use_atom_level_cross_attn', False)

            if self.use_atom_level_cross_attn:
                # ATOM-LEVEL CROSS-ATTENTION VARIANT
                # Head 1: Original KGDiff affinity prediction
                # Uses ligand embeddings from protein-ligand complex graph (WITH interaction)
                self.original_kgdiff_head = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    ShiftedSoftplus(),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                )

                # Head 2: Atom-level cross-attention affinity prediction
                # Multi-head cross-attention: protein atoms (query) attend to ligand atoms (key/value)
                # Both from separate graphs (WITHOUT interaction)
                self.num_cross_attn_heads = 4
                self.cross_attn_head_dim = self.hidden_dim // self.num_cross_attn_heads

                # Query, Key, Value projections for cross-attention
                self.cross_attn_query = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.cross_attn_key = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.cross_attn_value = nn.Linear(self.hidden_dim, self.hidden_dim)

                # Output projection after attention
                self.cross_attn_output = nn.Linear(self.hidden_dim, self.hidden_dim)

                # Final MLP for affinity prediction from attended features
                self.cross_attn_affinity_head = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    ShiftedSoftplus(),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                )

                print(f"[INFO] Using DUAL-HEAD Sam-PL architecture (ATOM-LEVEL CROSS-ATTENTION):")
                print(f"  Head1: Original KGDiff (ligand from complex graph WITH interaction)")
                print(f"  Head2: Atom-level cross-attention ({self.num_cross_attn_heads} heads, ligand/protein WITHOUT interaction)")
            else:
                # CONCAT/ATTENTION VARIANTS
                # Head 1: Interaction affinity prediction (WITH protein-ligand interaction)
                # Uses concatenated ligand+protein embeddings from complex graph
                self.interaction_affinity_head = nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Concat: ligand + protein
                    ShiftedSoftplus(),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                )

                # Head 2: Two implementation variants
                # Check if we should use attention-based Head 2 (for backward compatibility with old checkpoints)
                self.use_attention_head2 = getattr(config, 'use_attention_head2', False)

                if self.use_attention_head2:
                    # VARIANT 1: Attention-based Head 2 (original training implementation)
                    # Query: protein embedding, Key/Value: ligand embedding
                    self.non_interaction_query = nn.Linear(self.hidden_dim, self.hidden_dim)
                    self.non_interaction_key = nn.Linear(self.hidden_dim, self.hidden_dim)
                    self.non_interaction_value = nn.Linear(self.hidden_dim, self.hidden_dim)
                    self.non_interaction_scale = np.sqrt(self.hidden_dim)

                    # Final MLP for affinity prediction after attention
                    self.non_interaction_affinity_head = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        ShiftedSoftplus(),
                        nn.Linear(self.hidden_dim, 1),
                        nn.Sigmoid()
                    )
                    print(f"[INFO] Using DUAL-HEAD Sam-PL architecture (ATTENTION-BASED Head2):")
                    print(f"  Head1: Interaction-based (ligand+protein concat from complex graph)")
                    print(f"  Head2: Attention-based (Query=protein, Key/Value=ligand)")
                else:
                    # VARIANT 2: Concatenation-based Head 2 (newer implementation)
                    # Uses concatenated ligand+protein embeddings from separate graphs
                    self.non_interaction_affinity_head = nn.Sequential(
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Concat: ligand + protein (both non-interaction)
                        ShiftedSoftplus(),
                        nn.Linear(self.hidden_dim, 1),
                        nn.Sigmoid()
                    )
                    print(f"[INFO] Using DUAL-HEAD Sam-PL architecture (CONCAT-BASED Head2):")
                    print(f"  Head1: Interaction-based (ligand+protein concat from complex graph)")
                    print(f"  Head2: Concatenation-based (ligand+protein concat from separate graphs)")

        # Separate BA prediction: single head with non-interaction based affinity prediction
        self.use_separate_ba = getattr(config, 'use_separate_ba', False)
        if self.use_separate_ba:
            # Single head: Non-interaction affinity prediction (protein and ligand processed separately)
            self.separate_ba_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            print(f"[INFO] Using SEPARATE BA prediction: protein and ligand processed separately then concatenated")

        # Bidirectional atom-by-atom cross-attention
        self.use_bidirectional_cross_attn = getattr(config, 'use_bidirectional_cross_attn', False)
        if self.use_bidirectional_cross_attn:
            # Head 1: Interaction-based affinity prediction (WITH protein-ligand interaction)
            self.bidirectional_head1 = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Concat: ligand + protein
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )

            # Head 2: Bidirectional atom-level cross-attention (WITHOUT interaction)
            # Protein → Ligand attention
            self.bidirectional_p2l_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bidirectional_p2l_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bidirectional_p2l_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bidirectional_p2l_output = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Ligand → Protein attention
            self.bidirectional_l2p_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bidirectional_l2p_key = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bidirectional_l2p_value = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bidirectional_l2p_output = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Final MLP for affinity prediction from bidirectional attended features
            # Input: P(1*128) + L(1*128) = 256
            self.bidirectional_head2 = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )

            print(f"[INFO] Using BIDIRECTIONAL ATOM-BY-ATOM CROSS-ATTENTION:")
            print(f"  Head1: Interaction-based (ligand+protein concat from complex graph)")
            print(f"  Head2: Bidirectional atom-level cross-attention (P→L and L→P)")


## FOWARD ###
    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                protein_id, time_step=None, return_all=False, fix_x=False):
        """
        새로운 구현: UniTransformer 들어가기 전에 별도 그래프 구성
        - On-target: 단백질-리간드 복합체 그래프
        - Off-target: 단백질만의 그래프 (첫 번째 off-target만 사용)
        """
        batch_size = batch_protein.max().item() + 1

        # 리간드 특성 처리
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError("Invalid ligand feature shape")

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        # 원자 임베딩
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        # Node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === 1. On-target: 단백질-리간드 복합체 그래프 ===
        on_target_mask = (protein_id == 0)
        h_protein_on = h_protein[on_target_mask]
        pos_protein_on = protein_pos[on_target_mask]
        batch_protein_on = batch_protein[on_target_mask]

        # On-target 복합체 그래프 구성 (단백질 + 리간드)
        h_on_all, pos_on_all, batch_on_all, mask_ligand_on, mask_protein_on, _ = compose_context(
            h_protein=h_protein_on,
            h_ligand=init_ligand_h,
            pos_protein=pos_protein_on,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein_on,
            batch_ligand=batch_ligand,
            protein_id=None  # On-target은 단일 그래프이므로 protein_id 불필요
        )

        # On-target 복합체 그래프 처리 (UniTransformer)
        outputs_on = self.refine_net(h_on_all, pos_on_all, mask_ligand_on, batch_on_all,
                                   protein_id=None, return_all=False, fix_x=fix_x)

        # === 2. Off-target: 첫 번째 off-target 단백질만의 그래프 ===
        # protein_id == 1인 첫 번째 off-target만 선택
        off_target_mask = (protein_id == 1)

        # Check if off-target data exists
        if off_target_mask.any():
            h_protein_off = h_protein[off_target_mask]
            pos_protein_off = protein_pos[off_target_mask]
            batch_protein_off = batch_protein[off_target_mask]

            # Off-target 단백질만의 그래프 처리 (리간드 없음)
            mask_ligand_off = torch.zeros(len(h_protein_off), dtype=torch.bool, device=h_protein_off.device)
            outputs_off = self.refine_net(h_protein_off, pos_protein_off, mask_ligand_off, batch_protein_off,
                                        protein_id=None, return_all=False, fix_x=fix_x)

            # Off-target 단백질 임베딩 (scatter_mean으로 aggregation)
            off_target_protein_emb = scatter_mean(outputs_off['h'], batch_protein_off, dim=0)
        else:
            # No off-target data available, set to None or dummy embedding
            off_target_protein_emb = None

        # === 3. 결과 구성 ===
        # On-target에서 리간드 부분 추출
        final_ligand_h = outputs_on['h'][mask_ligand_on]
        final_ligand_pos = outputs_on['x'][mask_ligand_on]
        final_ligand_v = self.v_inference(final_ligand_h)

        # On-target 복합체 임베딩 (scatter_mean으로 aggregation)
        on_target_complex_emb = scatter_mean(outputs_on['h'], batch_on_all, dim=0)

        # === 4. Dual-head BA 예측 ===
        # 리간드 임베딩 집계 (On-target 복합체에서 나온 것)
        h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]

        # On-target BA 예측: 리간드 + 복합체 단백질 임베딩
        # On-target에서 단백질 부분만 추출
        on_protein_h = outputs_on['h'][mask_protein_on]
        batch_on_protein = batch_on_all[mask_protein_on]
        h_mol_on_protein = scatter_mean(on_protein_h, batch_on_protein, dim=0)

        # Handle different expert_pred architectures
        if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
            # ORIGINAL architecture: expert_pred takes only hidden_dim input
            # Use only ligand embedding (or could use only protein embedding)
            v_on_pred = self.expert_pred(h_mol_ligand)  # [batch_size, 1]

            if off_target_protein_emb is not None:
                # For off-target, could use ligand embedding (no protein interaction)
                v_off_pred = self.expert_pred(h_mol_ligand)  # [batch_size, 1]
            else:
                # No off-target data available, use same as on-target
                v_off_pred = v_on_pred.clone()
        else:
            # CURRENT architecture: expert_pred takes hidden_dim*2 input (ligand + protein)
            combined_on_emb = torch.cat([h_mol_ligand, h_mol_on_protein], dim=1)
            v_on_pred = self.expert_pred(combined_on_emb)  # [batch_size, 1]

            # Off-target BA 예측: 리간드 + 독립 단백질 임베딩 결합
            if off_target_protein_emb is not None:
                combined_off_emb = torch.cat([h_mol_ligand, off_target_protein_emb], dim=1)
                v_off_pred = self.expert_pred(combined_off_emb)  # [batch_size, 1]
            else:
                # No off-target data available, use dummy prediction (same as on-target for compatibility)
                v_off_pred = v_on_pred.clone()  # Use on-target prediction as fallback

        # === 원본 호환성을 위한 반환 구조 ===
        preds = {
            'ligand_pos': final_ligand_pos,
            'ligand_v': final_ligand_v,
            'ligand_h': final_ligand_h,
            'final_ligand_h': final_ligand_h,  # Add alias for joint_sequential_selectivity_guide
            'protein_h': outputs_on['h'],
            'final_h': outputs_on['h'],  # Add alias for joint_sequential_selectivity_guide
            'mask_ligand': mask_ligand_on,  # Add mask for joint_sequential_selectivity_guide
            'v_on_pred': v_on_pred.squeeze(-1),
            'v_off_pred': v_off_pred.squeeze(-1),
            'batch_ligand': batch_ligand
        }

        if return_all:
            preds.update({
                'on_target_complex_emb': on_target_complex_emb,
                'off_target_protein_emb': off_target_protein_emb,
                'off_target_protein_h': outputs_off['h']
            })

        return preds

    def forward_no_off_interaction(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                                   protein_id, time_step=None, return_all=False, fix_x=False):
        """
        NEW: Forward pass ensuring NO interaction between off-target protein and ligand
        - On-target: protein-ligand complex graph (WITH interaction)
        - Off-target: protein-only graph (NO ligand, NO interaction)
        - Off-target BA prediction: uses on-target ligand embedding + off-target protein embedding

        Key differences from forward():
        - Off-target protein processed completely separately (no ligand involved)
        - Off-target only uses its own atom embeddings and internal distances
        - No protein-ligand edges, no cross-attention, no 3D interaction for off-target
        """
        batch_size = batch_protein.max().item() + 1

        # Process ligand features
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError("Invalid ligand feature shape")

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        # Atom embeddings
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        # Node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === 1. On-target: protein-ligand complex graph (WITH INTERACTION) ===
        on_target_mask = (protein_id == 0)
        h_protein_on = h_protein[on_target_mask]
        pos_protein_on = protein_pos[on_target_mask]
        batch_protein_on = batch_protein[on_target_mask]

        # On-target complex graph (protein + ligand WITH interaction)
        h_on_all, pos_on_all, batch_on_all, mask_ligand_on, mask_protein_on, _ = compose_context(
            h_protein=h_protein_on,
            h_ligand=init_ligand_h,
            pos_protein=pos_protein_on,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein_on,
            batch_ligand=batch_ligand,
            protein_id=None
        )

        # Process on-target complex through UniTransformer (WITH protein-ligand interaction)
        outputs_on = self.refine_net(h_on_all, pos_on_all, mask_ligand_on, batch_on_all,
                                   protein_id=None, return_all=False, fix_x=fix_x)

        # === 2. Off-target: protein-only graph (NO LIGAND, NO INTERACTION) ===
        off_target_mask = (protein_id == 1)

        # Check if off-target data exists
        if off_target_mask.any():
            h_protein_off = h_protein[off_target_mask]
            pos_protein_off = protein_pos[off_target_mask]
            batch_protein_off = batch_protein[off_target_mask]

            # Process off-target protein ONLY (NO ligand involved)
            # mask_ligand_off = all False = all nodes are protein
            mask_ligand_off = torch.zeros(len(h_protein_off), dtype=torch.bool, device=h_protein_off.device)

            # RefineNet will ONLY create protein-protein edges (NO protein-ligand edges)
            outputs_off = self.refine_net(h_protein_off, pos_protein_off, mask_ligand_off, batch_protein_off,
                                        protein_id=None, return_all=False, fix_x=fix_x)

            # Off-target protein embedding (aggregated, NO ligand interaction)
            off_target_protein_emb = scatter_mean(outputs_off['h'], batch_protein_off, dim=0)
        else:
            # No off-target data available
            off_target_protein_emb = None

        # === 3. Extract results ===
        # Extract ligand from on-target complex (WITH interaction)
        final_ligand_h = outputs_on['h'][mask_ligand_on]
        final_ligand_pos = outputs_on['x'][mask_ligand_on]
        final_ligand_v = self.v_inference(final_ligand_h)

        # On-target complex embedding
        on_target_complex_emb = scatter_mean(outputs_on['h'], batch_on_all, dim=0)

        # === 4. Binding affinity predictions ===
        # Aggregate ligand embedding (from on-target WITH interaction)
        h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]

        # On-target BA prediction: ligand + on-target protein (both WITH interaction)
        on_protein_h = outputs_on['h'][mask_protein_on]
        batch_on_protein = batch_on_all[mask_protein_on]
        h_mol_on_protein = scatter_mean(on_protein_h, batch_on_protein, dim=0)

        # Handle different expert_pred architectures
        if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
            # ORIGINAL architecture: only ligand features
            v_on_pred = self.expert_pred(h_mol_ligand)  # [batch_size, 1]

            if off_target_protein_emb is not None:
                v_off_pred = self.expert_pred(h_mol_ligand)  # [batch_size, 1]
            else:
                v_off_pred = v_on_pred.clone()
        else:
            # CURRENT architecture: ligand + protein features
            combined_on_emb = torch.cat([h_mol_ligand, h_mol_on_protein], dim=1)
            v_on_pred = self.expert_pred(combined_on_emb)  # [batch_size, 1]

            # Off-target BA prediction:
            # - ligand embedding from on-target (WITH interaction)
            # - protein embedding from off-target (NO interaction with ligand)
            if off_target_protein_emb is not None:
                combined_off_emb = torch.cat([h_mol_ligand, off_target_protein_emb], dim=1)
                v_off_pred = self.expert_pred(combined_off_emb)  # [batch_size, 1]
            else:
                v_off_pred = v_on_pred.clone()

        # === Return structure ===
        preds = {
            'ligand_pos': final_ligand_pos,
            'ligand_v': final_ligand_v,
            'ligand_h': final_ligand_h,
            'final_ligand_h': final_ligand_h,  # For guidance functions
            'protein_h': outputs_on['h'],
            'final_h': outputs_on['h'],  # For guidance functions
            'mask_ligand': mask_ligand_on,
            'v_on_pred': v_on_pred.squeeze(-1),
            'v_off_pred': v_off_pred.squeeze(-1),
            'batch_ligand': batch_ligand
        }

        if return_all:
            preds.update({
                'on_target_complex_emb': on_target_complex_emb,
                'off_target_protein_emb': off_target_protein_emb,
                'off_target_protein_h': outputs_off['h'] if off_target_mask.any() else None
            })

        return preds

    def forward_sam_pl(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                       time_step=None, return_all=False, fix_x=False):
        """
        Sam-PL Forward: Two separate heads for interaction vs non-interaction affinity prediction
        1) Interaction head: protein-ligand complex graph (with edges between protein and ligand)
        2) Non-interaction head: separate protein and ligand graphs (no edges between them)
        """
        batch_size = batch_protein.max().item() + 1

        # Process ligand features
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError("Invalid ligand feature shape")

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        # Atom embeddings
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        # Node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === 1. Interaction-based head: protein-ligand complex graph ===
        h_complex, pos_complex, batch_complex, mask_ligand_complex, mask_protein_complex, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            protein_id=None
        )

        # Process through UniTransformer with interactions
        outputs_interaction = self.refine_net(h_complex, pos_complex, mask_ligand_complex, batch_complex,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Extract ligand results for generation (from interaction graph)
        final_ligand_pos = outputs_interaction['x'][mask_ligand_complex]
        final_ligand_h = outputs_interaction['h'][mask_ligand_complex]
        final_ligand_v = self.v_inference(final_ligand_h)

        # Get interaction-based embeddings
        h_mol_ligand_interaction = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]
        h_mol_protein_interaction = scatter_mean(outputs_interaction['h'][mask_protein_complex],
                                                batch_complex[mask_protein_complex], dim=0)  # [batch_size, hidden_dim]

        # === 2. Non-interaction head: separate protein and ligand graphs ===
        # Process protein separately (no ligand interaction)
        mask_protein_only = torch.zeros(len(h_protein), dtype=torch.bool, device=h_protein.device)
        outputs_protein_only = self.refine_net(h_protein, protein_pos, mask_protein_only, batch_protein,
                                              protein_id=None, return_all=False, fix_x=fix_x)

        # Process ligand separately (no protein interaction)
        mask_ligand_only = torch.ones(len(init_ligand_h), dtype=torch.bool, device=init_ligand_h.device)
        outputs_ligand_only = self.refine_net(init_ligand_h, init_ligand_pos, mask_ligand_only, batch_ligand,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Get non-interaction embeddings
        h_mol_ligand_non_interaction = scatter_mean(outputs_ligand_only['h'], batch_ligand, dim=0)  # [batch_size, hidden_dim]
        h_mol_protein_non_interaction = scatter_mean(outputs_protein_only['h'], batch_protein, dim=0)  # [batch_size, hidden_dim]

        # === 3. Dual affinity predictions ===
        # Head 1: Check architecture type
        if hasattr(self, 'original_kgdiff_head'):
            # Atom-level cross-attention architecture
            # Head 1: Original KGDiff affinity prediction
            # Use ligand embeddings from complex graph (WITH protein-ligand interaction)
            v_original_kgdiff_pred = self.original_kgdiff_head(h_mol_ligand_interaction).squeeze(-1)
        elif hasattr(self, 'interaction_affinity_head'):
            # Concat-based architecture
            # Concat ligand+protein embeddings from complex graph
            h_mol_interaction_concat = torch.cat([h_mol_ligand_interaction, h_mol_protein_interaction], dim=1)
            v_original_kgdiff_pred = self.interaction_affinity_head(h_mol_interaction_concat).squeeze(-1)
        else:
            # Fallback: use basic prediction
            v_original_kgdiff_pred = torch.zeros(h_mol_ligand_interaction.size(0), device=h_mol_ligand_interaction.device)

        # Head 2: Check architecture type
        if hasattr(self, 'cross_attn_query'):
            # Atom-level cross-attention architecture
            # Head 2: Atom-level cross-attention affinity prediction
            # Query: protein atoms from protein-only graph [num_protein_atoms, hidden_dim]
            # Key/Value: ligand atoms from ligand-only graph [num_ligand_atoms, hidden_dim]
            protein_h_atom = outputs_protein_only['h']  # [num_protein_atoms, hidden_dim]
            ligand_h_atom = outputs_ligand_only['h']    # [num_ligand_atoms, hidden_dim]

            # Project to Q, K, V
            Q = self.cross_attn_query(protein_h_atom)  # [num_protein_atoms, hidden_dim]
            K = self.cross_attn_key(ligand_h_atom)      # [num_ligand_atoms, hidden_dim]
            V = self.cross_attn_value(ligand_h_atom)    # [num_ligand_atoms, hidden_dim]

            # Reshape for multi-head attention
            batch_size_max = batch_protein.max().item() + 1
            Q = Q.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)  # [num_protein_atoms, num_heads, head_dim]
            K = K.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)  # [num_ligand_atoms, num_heads, head_dim]
            V = V.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)  # [num_ligand_atoms, num_heads, head_dim]

            # Compute attention per batch graph
            attended_protein_h = []
            for b in range(batch_size_max):
                protein_mask_b = batch_protein == b
                ligand_mask_b = batch_ligand == b

                Q_b = Q[protein_mask_b]  # [num_protein_atoms_b, num_heads, head_dim]
                K_b = K[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]
                V_b = V[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]

                # Compute attention: Q @ K^T / sqrt(d_k)
                attn_scores = torch.einsum('phd,lhd->phl', Q_b, K_b) / np.sqrt(self.cross_attn_head_dim)
                attn_weights = torch.softmax(attn_scores, dim=-1)  # [num_protein_atoms_b, num_heads, num_ligand_atoms_b]

                # Apply attention to values: attn @ V
                attended = torch.einsum('phl,lhd->phd', attn_weights, V_b)  # [num_protein_atoms_b, num_heads, head_dim]
                attended = attended.reshape(-1, self.hidden_dim)  # [num_protein_atoms_b, hidden_dim]

                attended_protein_h.append(attended)

            # Concatenate all batches
            attended_protein_h = torch.cat(attended_protein_h, dim=0)  # [num_protein_atoms, hidden_dim]

            # Output projection
            attended_protein_h = self.cross_attn_output(attended_protein_h)  # [num_protein_atoms, hidden_dim]

            # Aggregate to per-graph level
            h_mol_attended = scatter_mean(attended_protein_h, batch_protein, dim=0)  # [batch_size, hidden_dim]

            # Predict affinity from attended features
            v_cross_attn_pred = self.cross_attn_affinity_head(h_mol_attended).squeeze(-1)
        elif hasattr(self, 'non_interaction_affinity_head'):
            # Check which variant of Head 2 we're using
            if hasattr(self, 'non_interaction_query'):
                # VARIANT 1: Attention-based Head 2
                # Query: protein embedding, Key/Value: ligand embedding
                query = self.non_interaction_query(h_mol_protein_non_interaction)  # [batch_size, hidden_dim]
                key = self.non_interaction_key(h_mol_ligand_non_interaction)        # [batch_size, hidden_dim]
                value = self.non_interaction_value(h_mol_ligand_non_interaction)    # [batch_size, hidden_dim]

                # Compute attention scores: Q * K^T / sqrt(d_k)
                attention_scores = torch.sum(query * key, dim=-1, keepdim=True) / self.non_interaction_scale  # [batch_size, 1]

                # For single query-key pairs (batch_size=1), use sigmoid instead of softmax
                # to avoid losing protein information
                if h_mol_protein_non_interaction.size(0) == 1:
                    attention_weights = torch.sigmoid(attention_scores)  # [batch_size, 1]
                else:
                    attention_weights = torch.softmax(attention_scores, dim=0)  # [batch_size, 1]

                # Combine protein (query) and ligand (value) information
                # attended_features incorporates both protein and ligand
                attended_features = attention_weights * value + (1 - attention_weights) * query  # [batch_size, hidden_dim]

                # Predict affinity from attended features
                v_cross_attn_pred = self.non_interaction_affinity_head(attended_features).squeeze(-1)
            else:
                # VARIANT 2: Concatenation-based Head 2
                # Concat ligand+protein embeddings from separate graphs
                h_mol_non_interaction_concat = torch.cat([h_mol_ligand_non_interaction, h_mol_protein_non_interaction], dim=1)
                v_cross_attn_pred = self.non_interaction_affinity_head(h_mol_non_interaction_concat).squeeze(-1)
        else:
            # Fallback
            v_cross_attn_pred = torch.zeros(h_mol_ligand_interaction.size(0), device=h_mol_ligand_interaction.device)

        # Return results (use interaction-based results for generation)
        preds = {
            'ligand_pos': final_ligand_pos,
            'ligand_v': final_ligand_v,
            'ligand_h': final_ligand_h,
            'protein_h': outputs_interaction['h'],
            'v_original_kgdiff_pred': v_original_kgdiff_pred,
            'v_cross_attn_pred': v_cross_attn_pred,
            'batch_ligand': batch_ligand
        }

        if return_all:
            preds.update({
                'h_mol_ligand_interaction': h_mol_ligand_interaction,
                'h_mol_protein_interaction': h_mol_protein_interaction,
                'h_mol_ligand_non_interaction': h_mol_ligand_non_interaction,
                'h_mol_protein_non_interaction': h_mol_protein_non_interaction,
            })

        return preds

    def forward_atom_cross_attention(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                                      time_step=None, return_all=False, fix_x=False):
        """
        New training forward method with atom-level cross-attention:

        Head 1: Protein-ligand interaction-based BA prediction
            - Uses RefineNet output from protein-ligand complex graph (WITH interaction)
            - Ligand embeddings: scatter_mean across all ligand atoms
            - Protein embeddings: scatter_mean across all protein atoms
            - Concatenate and pass through MLP for BA prediction

        Head 2: Atom-level cross-attention BA prediction (NO interaction)
            - Process protein and ligand separately (NO edges between them)
            - Protein atoms from protein-only graph
            - Ligand atoms from ligand-only graph
            - Cross-attention mechanism:
                1. Query: Scatter mean protein atom embeddings (1 * 128)
                2. Key, Value: All ligand atom embeddings (N_ligand * 128)
                3. Attention score: (1 * N_ligand)
                4. Attended output: (1 * 128)
                5. MLP -> Sigmoid -> BA prediction

        Args:
            protein_pos: [N_protein, 3] protein atom positions
            protein_v: [N_protein, protein_feat_dim] protein atom features
            batch_protein: [N_protein] batch indices for protein atoms
            init_ligand_pos: [N_ligand, 3] ligand atom positions
            init_ligand_v: [N_ligand] or [N_ligand, ligand_feat_dim] ligand atom features
            batch_ligand: [N_ligand] batch indices for ligand atoms
            time_step: diffusion timestep
            return_all: whether to return intermediate results
            fix_x: whether to fix positions during refinement

        Returns:
            preds: dictionary containing:
                - ligand_pos: refined ligand positions (from interaction graph)
                - ligand_v: predicted ligand atom types (from interaction graph)
                - ligand_h: ligand atom embeddings (from interaction graph)
                - protein_h: protein atom embeddings (from interaction graph)
                - v_head1_pred: BA prediction from Head 1 (interaction-based)
                - v_head2_pred: BA prediction from Head 2 (cross-attention)
                - batch_ligand: batch indices
        """
        batch_size = batch_protein.max().item() + 1

        # Process ligand features
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError("Invalid ligand feature shape")

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        # Atom embeddings
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        # Node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === Head 1: Protein-ligand interaction graph ===
        # Compose protein-ligand complex graph (WITH edges between protein and ligand)
        h_complex, pos_complex, batch_complex, mask_ligand_complex, mask_protein_complex, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            protein_id=None
        )

        # Process through RefineNet with protein-ligand interactions
        outputs_interaction = self.refine_net(h_complex, pos_complex, mask_ligand_complex, batch_complex,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Extract ligand results for generation (from interaction graph)
        final_ligand_h = outputs_interaction['h'][mask_ligand_complex]
        final_ligand_pos = outputs_interaction['x'][mask_ligand_complex]
        final_ligand_v = self.v_inference(final_ligand_h)

        # Get interaction-based embeddings for Head 1
        h_mol_ligand_interaction = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]
        h_mol_protein_interaction = scatter_mean(outputs_interaction['h'][mask_protein_complex],
                                                batch_complex[mask_protein_complex], dim=0)  # [batch_size, hidden_dim]

        # Head 1 BA prediction: concatenate ligand + protein embeddings from interaction graph
        h_head1_concat = torch.cat([h_mol_ligand_interaction, h_mol_protein_interaction], dim=1)  # [batch_size, hidden_dim*2]

        # Check which head to use for Head 1
        if hasattr(self, 'original_kgdiff_head'):
            # Use only ligand embeddings (original KGDiff style)
            v_head1_pred = self.original_kgdiff_head(h_mol_ligand_interaction).squeeze(-1)
        elif hasattr(self, 'interaction_affinity_head'):
            # Use concatenated embeddings
            v_head1_pred = self.interaction_affinity_head(h_head1_concat).squeeze(-1)
        elif hasattr(self, 'expert_pred'):
            # Fallback to expert_pred
            if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
                v_head1_pred = self.expert_pred(h_mol_ligand_interaction).squeeze(-1)
            else:
                v_head1_pred = self.expert_pred(h_head1_concat).squeeze(-1)
        else:
            raise ValueError("No head1 prediction layer found")

        # === Head 2: Atom-level cross-attention (NO interaction) ===
        # Process protein separately (no ligand)
        mask_protein_only = torch.zeros(len(h_protein), dtype=torch.bool, device=h_protein.device)
        outputs_protein_only = self.refine_net(h_protein, protein_pos, mask_protein_only, batch_protein,
                                              protein_id=None, return_all=False, fix_x=fix_x)

        # Process ligand separately (no protein)
        mask_ligand_only = torch.ones(len(init_ligand_h), dtype=torch.bool, device=init_ligand_h.device)
        outputs_ligand_only = self.refine_net(init_ligand_h, init_ligand_pos, mask_ligand_only, batch_ligand,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Get atom-level embeddings (NO interaction)
        protein_h_atom = outputs_protein_only['h']  # [num_protein_atoms, hidden_dim]
        ligand_h_atom = outputs_ligand_only['h']    # [num_ligand_atoms, hidden_dim]

        # Atom-level cross-attention per batch
        # Query: scatter_mean of protein atoms -> (1 * 128) per batch
        # Key, Value: all ligand atoms -> (N_ligand * 128) per batch
        head2_attended_features = []

        for b in range(batch_size):
            # Get atoms for this batch
            protein_mask_b = batch_protein == b
            ligand_mask_b = batch_ligand == b

            protein_h_b = protein_h_atom[protein_mask_b]  # [num_protein_atoms_b, hidden_dim]
            ligand_h_b = ligand_h_atom[ligand_mask_b]      # [num_ligand_atoms_b, hidden_dim]

            # Query: scatter_mean across protein atoms -> (1, hidden_dim)
            query_b = protein_h_b.mean(dim=0, keepdim=True)  # [1, hidden_dim]

            # Key, Value: all ligand atoms
            if hasattr(self, 'cross_attn_query'):
                # Use cross-attention layers if available
                Q_b = self.cross_attn_query(query_b)  # [1, hidden_dim]
                K_b = self.cross_attn_key(ligand_h_b)  # [num_ligand_atoms_b, hidden_dim]
                V_b = self.cross_attn_value(ligand_h_b)  # [num_ligand_atoms_b, hidden_dim]

                # Compute attention scores: Q @ K^T / sqrt(d_k)
                # Q_b: [1, hidden_dim], K_b: [num_ligand_atoms_b, hidden_dim]
                # attn_scores: [1, num_ligand_atoms_b]
                attn_scores = torch.matmul(Q_b, K_b.transpose(0, 1)) / np.sqrt(self.hidden_dim)
                attn_weights = torch.softmax(attn_scores, dim=-1)  # [1, num_ligand_atoms_b]

                # Apply attention to values: attn @ V
                # attn_weights: [1, num_ligand_atoms_b], V_b: [num_ligand_atoms_b, hidden_dim]
                # attended: [1, hidden_dim]
                attended_b = torch.matmul(attn_weights, V_b)  # [1, hidden_dim]

                # Output projection
                attended_b = self.cross_attn_output(attended_b)  # [1, hidden_dim]
            else:
                # Simple attention without projection layers
                # Q: protein mean, K/V: ligand atoms
                attn_scores = torch.matmul(query_b, ligand_h_b.transpose(0, 1)) / np.sqrt(self.hidden_dim)
                attn_weights = torch.softmax(attn_scores, dim=-1)  # [1, num_ligand_atoms_b]
                attended_b = torch.matmul(attn_weights, ligand_h_b)  # [1, hidden_dim]

            head2_attended_features.append(attended_b.squeeze(0))  # [hidden_dim]

        # Stack all batch features
        h_head2_attended = torch.stack(head2_attended_features, dim=0)  # [batch_size, hidden_dim]

        # Head 2 BA prediction: MLP -> Sigmoid
        if hasattr(self, 'cross_attn_affinity_head'):
            v_head2_pred = self.cross_attn_affinity_head(h_head2_attended).squeeze(-1)
        elif hasattr(self, 'non_interaction_affinity_head'):
            v_head2_pred = self.non_interaction_affinity_head(h_head2_attended).squeeze(-1)
        elif hasattr(self, 'expert_pred'):
            # Fallback to expert_pred
            if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
                v_head2_pred = self.expert_pred(h_head2_attended).squeeze(-1)
            else:
                # Need to create dummy protein embedding for concatenation
                h_mol_protein_non_interaction = scatter_mean(protein_h_atom, batch_protein, dim=0)
                h_head2_concat = torch.cat([h_head2_attended, h_mol_protein_non_interaction], dim=1)
                v_head2_pred = self.expert_pred(h_head2_concat).squeeze(-1)
        else:
            raise ValueError("No head2 prediction layer found")

        # Return results (use interaction-based results for generation)
        preds = {
            'ligand_pos': final_ligand_pos,
            'ligand_v': final_ligand_v,
            'ligand_h': final_ligand_h,
            'protein_h': outputs_interaction['h'],
            'v_head1_pred': v_head1_pred,  # Interaction-based prediction
            'v_head2_pred': v_head2_pred,  # Cross-attention prediction
            'batch_ligand': batch_ligand,
            'mask_ligand': mask_ligand_complex  # For compatibility
        }

        if return_all:
            preds.update({
                'h_mol_ligand_interaction': h_mol_ligand_interaction,
                'h_mol_protein_interaction': h_mol_protein_interaction,
                'h_head2_attended': h_head2_attended,
                'protein_h_atom_non_interaction': protein_h_atom,
                'ligand_h_atom_non_interaction': ligand_h_atom,
            })

        return preds

    def forward_bidirectional_cross_attention(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                                               time_step=None, return_all=False, fix_x=False):
        """
        Bidirectional atom-by-atom cross-attention forward method:

        Head 1: Protein-ligand interaction-based BA prediction
            - Uses RefineNet output from protein-ligand complex graph (WITH interaction)
            - Ligand embeddings: scatter_mean across all ligand atoms
            - Protein embeddings: scatter_mean across all protein atoms
            - Concatenate and pass through MLP for BA prediction

        Head 2: Bidirectional atom-level cross-attention BA prediction (NO interaction)
            - Process protein and ligand separately (NO edges between them)
            - Protein→Ligand attention:
                1. Q: Protein atoms [N_protein, 128]
                2. K, V: Ligand atoms [N_ligand, 128]
                3. Attention: [N_protein, N_ligand]
                4. Output: [N_protein, 128] → Scatter mean → P[1, 128]
            - Ligand→Protein attention:
                5. Q: Ligand atoms [N_ligand, 128]
                6. K, V: Protein atoms [N_protein, 128]
                7. Attention: [N_ligand, N_protein]
                8. Output: [N_ligand, 128] → Scatter mean → L[1, 128]
            - Combine: P[1, 128] + L[1, 128] → [1, 256] → MLP → BA

        Args:
            protein_pos: [N_protein, 3] protein atom positions
            protein_v: [N_protein, protein_feat_dim] protein atom features
            batch_protein: [N_protein] batch indices for protein atoms
            init_ligand_pos: [N_ligand, 3] ligand atom positions
            init_ligand_v: [N_ligand] or [N_ligand, ligand_feat_dim] ligand atom features
            batch_ligand: [N_ligand] batch indices for ligand atoms
            time_step: diffusion timestep
            return_all: whether to return intermediate results
            fix_x: whether to fix positions during refinement

        Returns:
            preds: dictionary containing:
                - ligand_pos: refined ligand positions (from interaction graph)
                - ligand_v: predicted ligand atom types (from interaction graph)
                - ligand_h: ligand atom embeddings (from interaction graph)
                - protein_h: protein atom embeddings (from interaction graph)
                - v_head1_pred: BA prediction from Head 1 (interaction-based)
                - v_head2_pred: BA prediction from Head 2 (bidirectional cross-attention)
                - batch_ligand: batch indices
        """
        batch_size = batch_protein.max().item() + 1

        # Process ligand features
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError("Invalid ligand feature shape")

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        # Atom embeddings
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        # Node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === Head 1: Protein-ligand interaction graph ===
        # Compose protein-ligand complex graph (WITH edges between protein and ligand)
        h_complex, pos_complex, batch_complex, mask_ligand_complex, mask_protein_complex, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            protein_id=None
        )

        # Process through RefineNet with protein-ligand interactions
        outputs_interaction = self.refine_net(h_complex, pos_complex, mask_ligand_complex, batch_complex,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Extract ligand results for generation (from interaction graph)
        final_ligand_h = outputs_interaction['h'][mask_ligand_complex]
        final_ligand_pos = outputs_interaction['x'][mask_ligand_complex]
        final_ligand_v = self.v_inference(final_ligand_h)

        # Get interaction-based embeddings for Head 1
        h_mol_ligand_interaction = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]
        h_mol_protein_interaction = scatter_mean(outputs_interaction['h'][mask_protein_complex],
                                                batch_complex[mask_protein_complex], dim=0)  # [batch_size, hidden_dim]

        # Head 1 BA prediction: concatenate ligand + protein embeddings from interaction graph
        h_head1_concat = torch.cat([h_mol_ligand_interaction, h_mol_protein_interaction], dim=1)  # [batch_size, hidden_dim*2]

        # Use bidirectional_head1 if available, otherwise fallback
        if hasattr(self, 'bidirectional_head1'):
            v_head1_pred = self.bidirectional_head1(h_head1_concat).squeeze(-1)
        elif hasattr(self, 'interaction_affinity_head'):
            v_head1_pred = self.interaction_affinity_head(h_head1_concat).squeeze(-1)
        elif hasattr(self, 'expert_pred'):
            if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
                v_head1_pred = self.expert_pred(h_mol_ligand_interaction).squeeze(-1)
            else:
                v_head1_pred = self.expert_pred(h_head1_concat).squeeze(-1)
        else:
            raise ValueError("No head1 prediction layer found")

        # === Head 2: Bidirectional atom-level cross-attention (NO interaction) ===
        # Process protein separately (no ligand)
        mask_protein_only = torch.zeros(len(h_protein), dtype=torch.bool, device=h_protein.device)
        outputs_protein_only = self.refine_net(h_protein, protein_pos, mask_protein_only, batch_protein,
                                              protein_id=None, return_all=False, fix_x=fix_x)

        # Process ligand separately (no protein)
        mask_ligand_only = torch.ones(len(init_ligand_h), dtype=torch.bool, device=init_ligand_h.device)
        outputs_ligand_only = self.refine_net(init_ligand_h, init_ligand_pos, mask_ligand_only, batch_ligand,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Get atom-level embeddings (NO interaction)
        protein_h_atom = outputs_protein_only['h']  # [num_protein_atoms, hidden_dim]
        ligand_h_atom = outputs_ligand_only['h']    # [num_ligand_atoms, hidden_dim]

        # Bidirectional attention per batch
        p_attended_list = []  # P→L attended protein features
        l_attended_list = []  # L→P attended ligand features

        for b in range(batch_size):
            # Get atoms for this batch
            protein_mask_b = batch_protein == b
            ligand_mask_b = batch_ligand == b

            protein_h_b = protein_h_atom[protein_mask_b]  # [num_protein_atoms_b, hidden_dim]
            ligand_h_b = ligand_h_atom[ligand_mask_b]      # [num_ligand_atoms_b, hidden_dim]

            # === 1. Protein→Ligand Attention ===
            # Q: Protein atoms [N_protein, 128]
            # K, V: Ligand atoms [N_ligand, 128]
            if hasattr(self, 'bidirectional_p2l_query'):
                Q_p2l = self.bidirectional_p2l_query(protein_h_b)  # [num_protein_atoms_b, hidden_dim]
                K_p2l = self.bidirectional_p2l_key(ligand_h_b)      # [num_ligand_atoms_b, hidden_dim]
                V_p2l = self.bidirectional_p2l_value(ligand_h_b)    # [num_ligand_atoms_b, hidden_dim]

                # Attention: Q @ K^T / sqrt(d_k)
                # [num_protein_atoms_b, hidden_dim] @ [hidden_dim, num_ligand_atoms_b]
                # = [num_protein_atoms_b, num_ligand_atoms_b]
                attn_scores_p2l = torch.matmul(Q_p2l, K_p2l.transpose(0, 1)) / np.sqrt(self.hidden_dim)
                attn_weights_p2l = torch.softmax(attn_scores_p2l, dim=-1)  # [num_protein_atoms_b, num_ligand_atoms_b]

                # Apply attention to values
                # [num_protein_atoms_b, num_ligand_atoms_b] @ [num_ligand_atoms_b, hidden_dim]
                # = [num_protein_atoms_b, hidden_dim]
                attended_protein_b = torch.matmul(attn_weights_p2l, V_p2l)

                # Output projection
                attended_protein_b = self.bidirectional_p2l_output(attended_protein_b)  # [num_protein_atoms_b, hidden_dim]
            else:
                # Fallback: simple attention without projection
                attn_scores_p2l = torch.matmul(protein_h_b, ligand_h_b.transpose(0, 1)) / np.sqrt(self.hidden_dim)
                attn_weights_p2l = torch.softmax(attn_scores_p2l, dim=-1)
                attended_protein_b = torch.matmul(attn_weights_p2l, ligand_h_b)

            # Scatter mean to get per-graph protein representation: [1, hidden_dim]
            p_attended_b = attended_protein_b.mean(dim=0)  # [hidden_dim]
            p_attended_list.append(p_attended_b)

            # === 2. Ligand→Protein Attention ===
            # Q: Ligand atoms [N_ligand, 128]
            # K, V: Protein atoms [N_protein, 128]
            if hasattr(self, 'bidirectional_l2p_query'):
                Q_l2p = self.bidirectional_l2p_query(ligand_h_b)    # [num_ligand_atoms_b, hidden_dim]
                K_l2p = self.bidirectional_l2p_key(protein_h_b)      # [num_protein_atoms_b, hidden_dim]
                V_l2p = self.bidirectional_l2p_value(protein_h_b)    # [num_protein_atoms_b, hidden_dim]

                # Attention: Q @ K^T / sqrt(d_k)
                # [num_ligand_atoms_b, hidden_dim] @ [hidden_dim, num_protein_atoms_b]
                # = [num_ligand_atoms_b, num_protein_atoms_b]
                attn_scores_l2p = torch.matmul(Q_l2p, K_l2p.transpose(0, 1)) / np.sqrt(self.hidden_dim)
                attn_weights_l2p = torch.softmax(attn_scores_l2p, dim=-1)  # [num_ligand_atoms_b, num_protein_atoms_b]

                # Apply attention to values
                # [num_ligand_atoms_b, num_protein_atoms_b] @ [num_protein_atoms_b, hidden_dim]
                # = [num_ligand_atoms_b, hidden_dim]
                attended_ligand_b = torch.matmul(attn_weights_l2p, V_l2p)

                # Output projection
                attended_ligand_b = self.bidirectional_l2p_output(attended_ligand_b)  # [num_ligand_atoms_b, hidden_dim]
            else:
                # Fallback: simple attention without projection
                attn_scores_l2p = torch.matmul(ligand_h_b, protein_h_b.transpose(0, 1)) / np.sqrt(self.hidden_dim)
                attn_weights_l2p = torch.softmax(attn_scores_l2p, dim=-1)
                attended_ligand_b = torch.matmul(attn_weights_l2p, protein_h_b)

            # Scatter mean to get per-graph ligand representation: [1, hidden_dim]
            l_attended_b = attended_ligand_b.mean(dim=0)  # [hidden_dim]
            l_attended_list.append(l_attended_b)

        # Stack all batch features
        p_attended = torch.stack(p_attended_list, dim=0)  # [batch_size, hidden_dim]
        l_attended = torch.stack(l_attended_list, dim=0)  # [batch_size, hidden_dim]

        # Combine bidirectional features: P + L → [batch_size, hidden_dim*2]
        h_head2_bidirectional = torch.cat([p_attended, l_attended], dim=1)

        # Head 2 BA prediction: MLP → Sigmoid
        if hasattr(self, 'bidirectional_head2'):
            v_head2_pred = self.bidirectional_head2(h_head2_bidirectional).squeeze(-1)
        elif hasattr(self, 'non_interaction_affinity_head'):
            v_head2_pred = self.non_interaction_affinity_head(h_head2_bidirectional).squeeze(-1)
        elif hasattr(self, 'expert_pred'):
            if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim * 2:
                v_head2_pred = self.expert_pred(h_head2_bidirectional).squeeze(-1)
            else:
                # Use only one component
                v_head2_pred = self.expert_pred(p_attended).squeeze(-1)
        else:
            raise ValueError("No head2 prediction layer found")

        # Return results (use interaction-based results for generation)
        preds = {
            'ligand_pos': final_ligand_pos,
            'ligand_v': final_ligand_v,
            'ligand_h': final_ligand_h,
            'protein_h': outputs_interaction['h'],
            'v_head1_pred': v_head1_pred,  # Interaction-based prediction
            'v_head2_pred': v_head2_pred,  # Bidirectional cross-attention prediction
            'batch_ligand': batch_ligand,
            'mask_ligand': mask_ligand_complex  # For compatibility
        }

        if return_all:
            preds.update({
                'h_mol_ligand_interaction': h_mol_ligand_interaction,
                'h_mol_protein_interaction': h_mol_protein_interaction,
                'h_p_attended': p_attended,  # Protein→Ligand attended features
                'h_l_attended': l_attended,  # Ligand→Protein attended features
                'h_head2_bidirectional': h_head2_bidirectional,
                'protein_h_atom_non_interaction': protein_h_atom,
                'ligand_h_atom_non_interaction': ligand_h_atom,
            })

        return preds

    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, 
            protein_id, on_target_affinity, off_target_affinities, time_step=None
            ## 추가 :  On-target Affinity, Off-Target Affinities(3개) 추가
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode, center_ligand=self.center_ligand)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            protein_id=protein_id, #### 새로 추가
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['ligand_pos'], preds['ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        
        #loss_exp = F.mse_loss(preds['final_exp_pred'], affinity)
        #### 새로 추가 affinity loss
        # Ground-truth affinities are already normalized by trans.NormalizeVina transform
        # No need to normalize again - use directly
        on_target_normalized = on_target_affinity

        # Handle off_target_affinities reshaping
        batch_size = on_target_affinity.size(0)
        if off_target_affinities.dim() == 1:
            # Flattened structure: reshape to [batch_size, num_off_targets]
            num_off_targets = off_target_affinities.size(0) // batch_size
            off_target_normalized = off_target_affinities.view(batch_size, num_off_targets)
        else:
            off_target_normalized = off_target_affinities

        # 새로운 dual-head 구조에서 BA 예측 loss 계산
        if 'v_on_pred' in preds and 'v_off_pred' in preds:
            # On-target affinity loss
            loss_on = F.mse_loss(preds['v_on_pred'].squeeze(), on_target_normalized.squeeze())

            # Off-target affinity loss (첫 번째 off-target만 사용)
            off_target_first = off_target_normalized[:, 0]  # Always use first off-target
            loss_off = F.mse_loss(preds['v_off_pred'].squeeze(), off_target_first.squeeze())
        else:
            # Fallback: 기존 방식
            loss_on = F.mse_loss(preds['v_on_pred'].squeeze(), on_target_normalized.squeeze())

            # Handle dynamic off-target predictions
            off_pred = preds['v_off_pred']  # [batch_size, num_actual_off_targets]

            if off_pred.size(1) > 0:  # Has off-target predictions
                # Match ground truth size to predictions robustly
                batch_size = off_pred.size(0)
                num_off_preds = off_pred.size(1)

                # Ensure off_target_normalized has correct batch dimension
                if off_target_normalized.dim() == 1:
                    # Reshape 1D tensor to 2D: [total_elements] -> [batch_size, -1]
                    if off_target_normalized.numel() % batch_size == 0:
                        n_targets_per_batch = off_target_normalized.numel() // batch_size
                        off_target_reshaped = off_target_normalized.view(batch_size, n_targets_per_batch)
                    else:
                        # Fallback: repeat the tensor to match batch size
                        off_target_reshaped = off_target_normalized[:batch_size].unsqueeze(-1)
                else:
                    off_target_reshaped = off_target_normalized

                # Ensure batch dimension matches
                if off_target_reshaped.size(0) != batch_size:
                    # Adjust to correct batch size
                    if off_target_reshaped.size(0) > batch_size:
                        off_target_reshaped = off_target_reshaped[:batch_size]
                    else:
                        # Repeat to match batch size
                        repeat_factor = batch_size // off_target_reshaped.size(0)
                        remainder = batch_size % off_target_reshaped.size(0)
                        if remainder == 0:
                            off_target_reshaped = off_target_reshaped.repeat(repeat_factor, 1)
                        else:
                            off_target_reshaped = torch.cat([
                                off_target_reshaped.repeat(repeat_factor, 1),
                                off_target_reshaped[:remainder]
                            ], dim=0)

                # Match number of targets dimension
                if off_target_reshaped.size(1) >= num_off_preds:
                    off_gt_matched = off_target_reshaped[:, :num_off_preds]
                else:
                    # Pad with zeros if not enough targets
                    pad_size = num_off_preds - off_target_reshaped.size(1)
                    padding = torch.zeros(batch_size, pad_size, device=off_target_reshaped.device)
                    off_gt_matched = torch.cat([off_target_reshaped, padding], dim=1)

                loss_off = F.mse_loss(off_pred, off_gt_matched)
            else:
                # No off-target predictions
                loss_off = torch.tensor(0.0, device=on_target_normalized.device)
        
        loss_exp = self.lambda_on * loss_on + self.lambda_off * loss_off
        
        #if self.use_classifier_guide:
        loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight
        #else:
        #loss = loss_pos + loss_v * self.loss_v_weight
            
        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss_on_target_affinity': loss_on,
            'loss_off_target_affinity': loss_off,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_on_exp': preds['v_on_pred'], # For validation, we only care about on-target
            'normalized_on_target':on_target_normalized,
            'pred_off_exp': preds['v_off_pred'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['ligand_h']
        }

    def get_diffusion_loss_sam_pl(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
            on_target_affinity, off_target_affinities, time_step=None
    ):
        """
        Sam-PL diffusion loss with dual affinity heads:
        1) Interaction-based head (with protein-ligand interactions)
        2) Non-interaction-based head (without protein-ligand interactions)
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode, center_ligand=self.center_ligand)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN using Sam-PL dual head architecture
        # Check config flags to choose between forward methods
        use_simplified_cross_attn = getattr(self.config, 'use_simplified_cross_attn', False)
        use_bidirectional_cross_attn = getattr(self.config, 'use_bidirectional_cross_attn', False)

        if use_bidirectional_cross_attn:
            # Use bidirectional atom-by-atom cross-attention (forward_bidirectional_cross_attention)
            preds = self.forward_bidirectional_cross_attention(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos_perturbed,
                init_ligand_v=ligand_v_perturbed,
                batch_ligand=batch_ligand,
                time_step=time_step
            )
            # Normalize output keys for consistency
            # forward_bidirectional_cross_attention returns: v_head1_pred, v_head2_pred
            preds['v_original_kgdiff_pred'] = preds.get('v_head1_pred', preds.get('v_original_kgdiff_pred'))
            preds['v_cross_attn_pred'] = preds.get('v_head2_pred', preds.get('v_cross_attn_pred'))
        elif use_simplified_cross_attn:
            # Use simplified atom-level cross-attention (forward_atom_cross_attention)
            preds = self.forward_atom_cross_attention(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos_perturbed,
                init_ligand_v=ligand_v_perturbed,
                batch_ligand=batch_ligand,
                time_step=time_step
            )
            # Normalize output keys for consistency
            # forward_atom_cross_attention returns: v_head1_pred, v_head2_pred
            preds['v_original_kgdiff_pred'] = preds.get('v_head1_pred', preds.get('v_original_kgdiff_pred'))
            preds['v_cross_attn_pred'] = preds.get('v_head2_pred', preds.get('v_cross_attn_pred'))
        else:
            # Use original forward_sam_pl (per-atom cross-attention)
            preds = self.forward_sam_pl(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos_perturbed,
                init_ligand_v=ligand_v_perturbed,
                batch_ligand=batch_ligand,
                time_step=time_step
            )
            # forward_sam_pl already returns: v_original_kgdiff_pred, v_cross_attn_pred

        pred_ligand_pos, pred_ligand_v = preds['ligand_pos'], preds['ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # atom position loss
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # Sam-PL dual affinity loss
        on_target_normalized = on_target_affinity

        # Head 1: Original KGDiff affinity loss
        loss_original_kgdiff = F.mse_loss(preds['v_original_kgdiff_pred'].squeeze(), on_target_normalized.squeeze())

        # Head 2: Cross-attention affinity loss
        loss_cross_attn = F.mse_loss(preds['v_cross_attn_pred'].squeeze(), on_target_normalized.squeeze())

        # Combined affinity loss
        loss_exp = loss_original_kgdiff + loss_cross_attn

        # Total loss
        loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss_original_kgdiff': loss_original_kgdiff,
            'loss_cross_attn': loss_cross_attn,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_original_kgdiff_exp': preds['v_original_kgdiff_pred'],
            'pred_cross_attn_exp': preds['v_cross_attn_pred'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['ligand_h']
        }

    def dual_head_ba_prediction(self, final_h, final_ligand_h, mask_ligand, batch_protein, batch_ligand,
                               off_protein_h=None):
        """
        Dual-head binding affinity prediction for joint guide mode with single off-target.

        Args:
            final_h: Combined protein-ligand embeddings with interactions
            final_ligand_h: Ligand embeddings from interaction state
            mask_ligand: Mask for ligand atoms
            batch_protein: Protein batch indices
            batch_ligand: Ligand batch indices
            off_protein_h: Single off-target protein embeddings (without ligand interaction)

        Returns:
            Dictionary with on-target and off-target BA predictions
        """
        from torch_scatter import scatter_mean

        # Get ligand molecular representation (same for both heads)
        h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]

        # Head 1: On-target BA prediction (with protein-ligand interaction)
        h_mol_protein = scatter_mean(final_h[~mask_ligand], batch_protein, dim=0)  # [batch_size, hidden_dim]
        expert_input_on = torch.cat([h_mol_ligand, h_mol_protein], dim=1)  # [batch_size, hidden_dim * 2]
        pred_affinity_on = self.expert_pred_on(expert_input_on).squeeze(-1)  # [batch_size]

        # Head 2: Off-target BA prediction (without protein-ligand interaction)
        if off_protein_h is not None:
            # Aggregate off-target protein features
            h_mol_off_protein = scatter_mean(off_protein_h, batch_protein, dim=0)  # [batch_size, hidden_dim]

            # Ensure batch dimension consistency
            if h_mol_ligand.size(0) != h_mol_off_protein.size(0):
                # Expand to match batch size
                min_batch_size = min(h_mol_ligand.size(0), h_mol_off_protein.size(0))
                h_mol_ligand_matched = h_mol_ligand[:min_batch_size]
                h_mol_off_protein_matched = h_mol_off_protein[:min_batch_size]
            else:
                h_mol_ligand_matched = h_mol_ligand
                h_mol_off_protein_matched = h_mol_off_protein

            expert_input_off = torch.cat([h_mol_ligand_matched, h_mol_off_protein_matched], dim=1)
            pred_affinity_off = self.expert_pred_off(expert_input_off).squeeze(-1)  # [batch_size]
        else:
            # No off-target data, use dummy zeros
            pred_affinity_off = torch.zeros_like(pred_affinity_on)

        return {
            'pred_affinity_on': pred_affinity_on,
            'pred_affinity_off': pred_affinity_off,
            'h_mol_ligand': h_mol_ligand,
            'h_mol_protein': h_mol_protein
        }

    def calc_atom_dis(
            self, protein_pos, protein_v, affinity, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode, center_ligand=self.center_ligand)

        time_step_arr = torch.arange(0,1001,20).to(protein_pos.device)
        time_step_arr[-1] = 999
        lig_pro_dis_all = []
        for time_step in tqdm(time_step_arr):
            time_step = torch.tensor(time_step.tolist()).repeat(num_graphs).to(protein_pos.device)
            a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

            # 2. perturb pos and v
            a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
            pos_noise = torch.zeros_like(ligand_pos)
            pos_noise.normal_()
            # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
            ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
            # Vt = a * V0 + (1-a) / K
            lig_pro_dis = []
            for batch_idx in range(num_graphs):
                
                pro_coords = protein_pos[batch_protein==batch_idx]
                pro_ident = torch.tensor([0]).repeat(pro_coords.shape[0],1)
            #     print(loader_data)
                mol_coords = ligand_pos_perturbed[batch_ligand==batch_idx]
                mol_ident = torch.tensor([1]).repeat(mol_coords.shape[0],1)
                all_coords = torch.cat((mol_coords,pro_coords),dim=0)
                all_ident = torch.cat((mol_ident, pro_ident),dim=0)
                all_ident_rep = all_ident.T.repeat(all_coords.shape[0],1)
            #         all_coords_ident = torch.cat((all_coords, all_ident), dim=1)
                dist = torch.sum((all_coords[:,None,:] - all_coords[None,:,:])**2,dim=-1).sqrt()
                _, ind = torch.sort(dist, 1)
                # num_lig_atom = []
                num_lig_atom = all_ident[ind][:,:32].sum(dim=1)
                # for dis, indice in zip(all_ident_rep, ind):
                #     num_lig_atom.append(dis[indice][:32].sum(dim=0,keepdim=True))
                # num_lig_atom = torch.stack(num_lig_atom,dim=0)
                num_lig_atom_ident = torch.cat((all_ident, num_lig_atom),dim=1)
                lig_pro_dis.append(num_lig_atom_ident)
            
            lig_pro_dis = torch.cat(lig_pro_dis,dim=0)
            lig_pro_dis_all.append(lig_pro_dis)
        # lig_pro_dis_all = torch.cat(lig_pro_dis_all,dim=0)
        
        torch.save(lig_pro_dis_all, 'knn32_atom_type_num_across_1000step.pt')
    # def classifier_gradient(self, input, batch_all, t):
    #     with torch.enable_grad():
    #         x_in = input.detach().requires_grad_(True)
    #         final_exp_pred = scatter_mean(self.expert_pred(x_in).squeeze(-1), batch_all)
    #         grad = torch.autograd.grad(_exp_pred, x_in,grad_outputs=torch.ones_like(final_exp_pred))[0]
    #         return grad
        
    def pv_joint_guide(self, ligand_v_index, ligand_pos, protein_v, protein_pos, batch_protein, batch_ligand, protein_id, w_off=1.0, on_target_only=False):
        """
        Joint guidance with option for on-target only mode

        Args:
            w_off: Weight for off-target penalty (higher = stronger off-target avoidance)
            on_target_only: If True, use original KGDiff joint guidance (single protein)

        If on_target_only=True: Use original KGDiff joint guidance (single protein affinity)
        If on_target_only=False: S = v_on - w_off * mean(v_off) (selectivity score)
        """

        # Memory monitoring for selectivity mode (COMMENTED OUT)
        # def print_memory_usage(stage):
        #     if torch.cuda.is_available():
        #         allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        #         reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        #         free_reserved = reserved - allocated
        #         total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        #         print(f"[SELECTIVITY MEMORY] {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free_Reserved={free_reserved:.2f}GB, Total={total_gpu:.2f}GB")

        # print_memory_usage("START pv_joint_guide")
        
        if on_target_only:
            # ON-TARGET ONLY MODE: Use original KGDiff joint guidance (like molopt_score_model_original.py)
            with torch.enable_grad():
                ligand_v = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
                ligand_pos = ligand_pos.detach().requires_grad_(True)
                
                # Original KGDiff forward pass (single protein)
                init_h_protein = self.protein_atom_emb(protein_v)
                init_ligand_h = self.ligand_atom_emb(ligand_v)
                h_protein = torch.cat([init_h_protein, torch.zeros(len(init_h_protein), 1).to(init_h_protein)], -1)
                ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(init_ligand_h)], -1)

                # For on-target only mode, create dummy protein_id (all zeros for single protein)
                dummy_protein_id = torch.zeros_like(batch_protein)
                
                h_all, pos_all, batch_all, mask_ligand, mask_protein, protein_id_ctx = compose_context(
                    h_protein=h_protein,
                    h_ligand=ligand_h,
                    pos_protein=protein_pos,
                    pos_ligand=ligand_pos,
                    batch_protein=batch_protein,
                    batch_ligand=batch_ligand,
                    protein_id=dummy_protein_id,
                )

                outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all)
                final_pos, final_h = outputs['x'], outputs['h']
                final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
                
                # Affinity prediction (adapted for both architectures)
                # Get molecular representations
                h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]
                h_mol_protein = scatter_mean(final_h[~mask_ligand], batch_protein, dim=0)  # [batch_size, hidden_dim]
                
                # Create appropriate input based on architecture
                if getattr(self.config, 'use_dual_head_ba', False):
                    # Dual-head: predict both on-target and off-target BA
                    # First, we need to get off-target protein embeddings without ligand interaction

                    # Get unique protein IDs to find off-target proteins
                    unique_protein_ids = torch.unique(protein_id)

                    # In the dataset structure:
                    # protein_id = 0: on-target protein
                    # protein_id = 1,2,3...: off-target proteins
                    on_target_id = 0  # Explicit on-target ID

                    # Find first off-target protein (protein_id = 1)
                    off_target_id = None
                    for pid in unique_protein_ids:
                        if pid != 0:  # Any protein that's not on-target
                            off_target_id = pid
                            break

                    off_protein_h = None
                    if off_target_id is not None:
                        # Get off-target protein without ligand interaction
                        off_protein_mask = (protein_id == off_target_id)
                        off_protein_pos = protein_pos[off_protein_mask]
                        off_protein_v = protein_v[off_protein_mask]
                        off_batch_protein = batch_protein[off_protein_mask]

                        if len(off_protein_pos) == 0:
                            print(f"[Dual-Head] WARNING: Off-target protein (ID={off_target_id}) has no atoms!")
                            off_protein_h = None
                        else:
                            # Forward pass with only off-target protein (no ligand interaction)
                            with torch.no_grad():
                                off_init_h_protein = self.protein_atom_emb(off_protein_v)
                                off_h_protein = torch.cat([off_init_h_protein, torch.zeros(len(off_init_h_protein), 1).to(off_init_h_protein)], -1)

                                # Create context with only protein
                                off_h_all, off_pos_all, off_batch_all, _, _, _ = compose_context(
                                    h_protein=off_h_protein,
                                    h_ligand=torch.empty(0, off_h_protein.size(-1), device=off_h_protein.device),
                                    pos_protein=off_protein_pos,
                                    pos_ligand=torch.empty(0, 3, device=off_protein_pos.device),
                                    batch_protein=off_batch_protein,
                                    batch_ligand=torch.empty(0, dtype=torch.long, device=off_batch_protein.device),
                                    protein_id=torch.full_like(off_batch_protein, off_target_id),
                                )

                                # Forward pass to get off-target protein embeddings
                                off_outputs = self.refine_net(off_h_all, off_pos_all, torch.zeros_like(off_batch_all, dtype=torch.bool), off_batch_all)
                                off_protein_h = off_outputs['h']  # All are protein atoms since no ligand

                    # Use dual-head prediction
                    dual_results = self.dual_head_ba_prediction(
                        final_h=final_h,
                        final_ligand_h=final_ligand_h,
                        mask_ligand=mask_ligand,
                        batch_protein=batch_protein,
                        batch_ligand=batch_ligand,
                        off_protein_h=off_protein_h
                    )

                    pred_affinity = dual_results['pred_affinity_on']  # Use on-target for guidance
                    # Store dual results for loss calculation
                    self.dual_head_results = dual_results

                    # Debug info
                    if off_target_id is not None:
                        print(f"[Dual-Head] Using on-target (ID={on_target_id}) and off-target (ID={off_target_id})")
                    else:
                        print(f"[Dual-Head] Only on-target protein available (ID={on_target_id})")

                elif getattr(self.config, 'use_original_expert', False):
                    # Original: use ligand features only
                    expert_input = h_mol_ligand
                    pred_affinity = self.expert_pred(expert_input).squeeze(-1)  # [batch_size]
                else:
                    # Current: concatenate ligand + protein features
                    expert_input = torch.cat([h_mol_ligand, h_mol_protein], dim=1)  # [batch_size, hidden_dim * 2]
                    pred_affinity = self.expert_pred(expert_input).squeeze(-1)  # [batch_size]
                
                # For compatibility with original KGDiff, also compute atom-level affinity
                atom_affinity = torch.zeros_like(batch_ligand, dtype=torch.float, device=batch_ligand.device)
                for i in range(len(pred_affinity)):
                    ligand_mask_i = batch_ligand == i
                    atom_affinity[ligand_mask_i] = pred_affinity[i].float()  # Convert to float to match dtype
                
                pred_affinity_log = pred_affinity.log()
                
                # Original KGDiff gradients
                type_grad = torch.autograd.grad(pred_affinity, ligand_v, grad_outputs=torch.ones_like(pred_affinity), retain_graph=True)[0]
                pos_grad = torch.autograd.grad(pred_affinity_log, ligand_pos, grad_outputs=torch.ones_like(pred_affinity), retain_graph=True)[0]
            
            final_ligand_v = self.v_inference(final_ligand_h)

            preds = {
                'pred_ligand_pos': final_ligand_pos,
                'pred_ligand_v': final_ligand_v,
                'atom_affinity': atom_affinity,
                'final_h': final_h,
                'final_ligand_h': final_ligand_h,
                'final_exp_pred': pred_affinity,
                'batch_all': batch_all,
                'mask_ligand': mask_ligand,
            }
            return preds, type_grad, pos_grad
            
        else:
            # MULTI-PROTEIN SELECTIVITY MODE: Use the existing selectivity logic
            # print_memory_usage("BEFORE selectivity mode")

            # Aggressive memory clearing
            torch.cuda.empty_cache()
            gc.collect()
            # print_memory_usage("AFTER memory cleanup")
            
            # Store original device
            original_device = protein_pos.device
            
            try:
                # Disable mixed precision for better numerical stability in selectivity mode
                use_autocast = False  # Disabled to prevent NaN issues
                with torch.cuda.amp.autocast(enabled=use_autocast):
                    with torch.enable_grad():
                        # Prepare tensors for gradient computation (let autocast handle precision)
                        ligand_v = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
                        ligand_pos = ligand_pos.detach().requires_grad_(True)
                        
                        # Create time step (needed for forward pass)
                        batch_size = batch_protein.max().item() + 1
                        time_step = torch.zeros(batch_size, dtype=torch.long, device=protein_pos.device)
                        
                        # Forward pass
                        # print_memory_usage("BEFORE forward pass")
                        preds = self(
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            time_step=time_step,
                            protein_id=protein_id
                        )
                        # print_memory_usage("AFTER forward pass")
                        
                        # Extract predictions and convert to float for gradient computation
                        v_on_pred = preds['v_on_pred'].float()      # [batch_size, 1]
                        v_off_pred = preds['v_off_pred'].float()    # [batch_size, num_off_targets] or [batch_size]

                        # Selectivity mode: S = v_on - w_off * mean(v_off)
                        if v_off_pred.numel() > 0:
                            # Handle both squeezed [batch_size] and unsqueezed [batch_size, n] shapes
                            if v_off_pred.dim() == 1:
                                # Single off-target, already squeezed to [batch_size]
                                v_off_mean = v_off_pred.unsqueeze(1)  # [batch_size, 1]
                            else:
                                # Multiple off-targets [batch_size, num_off_targets]
                                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)  # [batch_size, 1]
                            guidance_score = v_on_pred - w_off * v_off_mean
                        else:
                            # Fallback if no off-targets
                            guidance_score = v_on_pred
                        
                        # Calculate gradients with respect to the guidance score
                        # IMPORTANT: Follow original KGDiff gradient computation style
                        # Type gradient: w.r.t. guidance_score (NOT log)
                        # Position gradient: w.r.t. log(guidance_score)
                        # print_memory_usage("BEFORE gradient calculation")
                        type_grad = torch.autograd.grad(
                            guidance_score.sum(), ligand_v,
                            retain_graph=True, create_graph=False
                        )[0]

                        guidance_score_log = torch.log(guidance_score + 1e-8)
                        pos_grad = torch.autograd.grad(
                            guidance_score_log.sum(), ligand_pos,
                            retain_graph=False, create_graph=False
                        )[0]
                        # print_memory_usage("AFTER gradient calculation")
                        
                        # Create final predictions dict
                        # IMPORTANT: Must include x0/v0 reconstruction from forward pass for posterior computation
                        final_preds = {
                            'v_on_pred': v_on_pred.detach(),
                            'v_off_pred': v_off_pred.detach() if v_off_pred.numel() > 0 else torch.empty(0),
                            'selectivity_score': guidance_score.detach(),
                            'final_exp_pred': guidance_score.detach(),  # For compatibility
                        }

                        # Add x0/v0 reconstruction from forward pass (required for posterior calculation)
                        # Forward pass returns 'ligand_pos' and 'ligand_v' as reconstructed x0/v0
                        if 'ligand_pos' in preds:
                            final_preds['pred_ligand_pos'] = preds['ligand_pos']
                        if 'ligand_v' in preds:
                            final_preds['pred_ligand_v'] = preds['ligand_v']
                        # Legacy keys for backward compatibility
                        if 'pred_ligand_pos' in preds:
                            final_preds['pred_ligand_pos'] = preds['pred_ligand_pos']
                        if 'pred_ligand_v' in preds:
                            final_preds['pred_ligand_v'] = preds['pred_ligand_v']

                        # print_memory_usage("END pv_joint_guide - SUCCESS")
                        return final_preds, type_grad.detach(), pos_grad.detach()

            except Exception as e:
                # Fallback to CPU computation or return error
                # print_memory_usage("ERROR in selectivity guidance")
                print(f"Error in selectivity guidance: {e}")
                import traceback
                traceback.print_exc()
                # Return dummy values to prevent crash
                dummy_preds = {
                    'pred_ligand_pos': torch.zeros_like(ligand_pos),
                    'pred_ligand_v': torch.zeros(len(ligand_pos), self.num_classes).to(ligand_pos.device),
                    'v_on_pred': torch.zeros(1),
                    'v_off_pred': torch.empty(0),
                    'selectivity_score': torch.zeros(1),
                    'final_exp_pred': torch.zeros(1),
                }
                # IMPORTANT: type_grad must match ligand_v shape [num_atoms, num_classes]
                # pos_grad must match ligand_pos shape [num_atoms, 3]
                dummy_type_grad = torch.zeros(len(ligand_pos), self.num_classes, device=ligand_pos.device)
                dummy_pos_grad = torch.zeros_like(ligand_pos)
                return dummy_preds, dummy_type_grad, dummy_pos_grad
    # def classifier_gradient(self, input, batch_all, t):
    #     with torch.enable_grad():
    #         x_in = input.detach().requires_grad_(True)
    #         final_exp_pred = scatter_mean(self.expert_pred(x_in).squeeze(-1), batch_all)
    #         grad = torch.autograd.grad(final_exp_pred, x_in,grad_outputs=torch.ones_like(final_exp_pred))[0]
    #         return grad

    def vina_classifier_gradient(self, logits_ligand_v_recon, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t):
        
        with torch.enable_grad():
            x_in = logits_ligand_v_recon.detach().requires_grad_(True)
            ligand_pos_in = ligand_pos.detach().requires_grad_(True)
            
            vina_score, vina_score_each = calc_vina(F.gumbel_softmax(x_in,hard=True,tau=0.5), ligand_pos_in, protein_v, protein_pos, batch_ligand, batch_protein)
            grad1 = torch.autograd.grad(vina_score, x_in,grad_outputs=torch.ones_like(vina_score), create_graph=True)[0]
            grad2 = torch.autograd.grad(vina_score, ligand_pos_in,grad_outputs=torch.ones_like(vina_score), create_graph=True)[0]
            return grad1, grad2, vina_score_each
        
    def value_net_classifier_gradient(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t, value_model):
        value_model.eval()
        with torch.enable_grad():
            
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob,hard=True,tau=0.5)
            preds = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            pred_affinity = preds['final_exp_pred']
            
            grad1 = torch.autograd.grad(pred_affinity, ligand_v_next_prob,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            grad2 = torch.autograd.grad(pred_affinity, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            return grad1, grad2, pred_affinity
        
    def value_net_classifier_gradient_rep(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t, value_model):
        value_model.eval()
        with torch.enable_grad():
            
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob,hard=True,tau=0.5)
            preds = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            pred_affinity = preds['final_exp_pred']
            
            grad1 = torch.autograd.grad(pred_affinity, ligand_v_next_prob,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            grad2 = torch.autograd.grad(pred_affinity, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            w = self.pos_classifier_grad_weight[t].to(ligand_v_next_prob.device)[0]
            return grad1 / torch.sqrt(w**2+1), grad2 / torch.sqrt(w**2+1), pred_affinity

    def value_net_classifier_gradient_rep2(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t, value_model):
        value_model.eval()
        with torch.enable_grad():
            
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob,hard=True,tau=0.5)
            preds = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            pred_affinity = preds['final_exp_pred']
            
            grad1 = torch.autograd.grad(pred_affinity, ligand_v_next_prob,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            grad2 = torch.autograd.grad(pred_affinity, ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            w2 = self.pos_classifier_grad_weight[t].to(ligand_v_next_prob.device)[0]
            w1 = self.log_alphas_v[t].exp().to(ligand_v_next_prob.device)[0]
            return grad1 * w1, grad2 / torch.sqrt(w2**2+1), pred_affinity
        
    def valuenet_sequential_selectivity_guide(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos,
                                            batch_protein, batch_ligand, off_target_data, t, value_model,
                                            w_on=1.0, w_off=1.0):
        """
        Valuenet-based TRUE sequential selectivity guidance
        - Step 1: On-target + ligand forward pass with value_model → get final_ligand_h
        - Step 2: Each off-target + ligand (using same final_ligand_h) → get v_off_pred
        - True sequential processing without information mixing
        """

        value_model.eval()

        with torch.enable_grad():
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob, hard=True, tau=0.5)

            # Step 1: Process on-target + ligand with value_model to get reference ligand embedding
            time_step_on = torch.zeros(1, dtype=torch.long, device=protein_pos.device)
            protein_id_on = torch.zeros_like(batch_protein)

            preds_on = value_model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                protein_id=protein_id_on,
                time_step=time_step_on
            )

            # Extract on-target predictions and ligand embedding
            v_on_pred = preds_on.get('v_on_pred', preds_on.get('final_exp_pred'))
            final_ligand_h = preds_on['final_ligand_h']  # Fixed ligand embedding from on-target

            # Step 2: Process each off-target sequentially with same ligand embedding
            v_off_pred_list = []

            # Handle case when off_target_data is empty (on-target only mode)
            if not off_target_data:
                # Use on-target prediction as off-target fallback
                # Ensure v_on_pred has correct dimension for concatenation
                if v_on_pred.dim() == 1:
                    v_off_pred_single = v_on_pred.unsqueeze(1)  # [batch_size, 1]
                else:
                    v_off_pred_single = v_on_pred
                v_off_pred_list = [v_off_pred_single]
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)
                    off_batch_protein = off_data.get('protein_element_batch',
                                                   torch.zeros_like(off_protein_pos[:, 0], dtype=torch.long)).to(protein_pos.device)

                # Create single protein_id for this off-target
                off_protein_id = torch.zeros_like(off_batch_protein)
                time_step_off = torch.zeros(1, dtype=torch.long, device=protein_pos.device)

                # Forward pass for this off-target + ligand with value_model
                preds_off = value_model(
                    protein_pos=off_protein_pos,
                    protein_v=off_protein_v,
                    batch_protein=off_batch_protein,
                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v_next,
                    batch_ligand=batch_ligand,
                    protein_id=off_protein_id,
                    time_step=time_step_off
                )

                # Extract off-target protein features and compute binding affinity
                # Use fixed final_ligand_h from on-target, not the new ligand embedding
                off_final_h = preds_off['final_h']
                off_mask_protein = preds_off['mask_ligand'] == False  # Protein mask
                off_protein_h = off_final_h[off_mask_protein]

                # Aggregate off-target protein features
                h_mol_off_target = off_protein_h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

                # Use fixed ligand embedding from on-target (properly aggregated by batch)
                from torch_scatter import scatter_mean
                h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]
                # Ensure matching batch dimensions with off-target protein
                # FIX: Expand off-target protein to match number of ligand samples (not the other way around!)
                if h_mol_ligand.size(0) != h_mol_off_target.size(0):
                    h_mol_off_target = h_mol_off_target[0:1].expand(h_mol_ligand.size(0), -1)

                # Predict binding affinity for this off-target using value_model's expert_pred
                # Check value_model's expert_pred input dimension and adapt accordingly
                if hasattr(value_model, 'expert_input_dim') and value_model.expert_input_dim == value_model.hidden_dim:
                    # Original architecture: only ligand features
                    expert_input_off = h_mol_ligand
                else:
                    # Current architecture: ligand + protein features
                    expert_input_off = torch.cat([h_mol_ligand, h_mol_off_target], dim=1)
                v_off_pred = value_model.expert_pred(expert_input_off)
                v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)  # [batch_size, num_off_targets]
                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)  # [batch_size, 1]
            else:
                v_off_pred = torch.empty(v_on_pred.shape[0], 0, device=v_on_pred.device)
                v_off_mean = torch.zeros_like(v_on_pred)

            # Compute selectivity score
            selectivity_score = w_on * v_on_pred - w_off * v_off_mean

            # Apply log for position gradient (like original KGDiff pv_joint_guide)
            # Type gradient: direct selectivity_score
            # Position gradient: log(selectivity_score) for better gradient scaling
            selectivity_score_log = selectivity_score.log()

            # Compute gradients with respect to selectivity score
            # Type gradient: w.r.t selectivity_score (not log)
            grad1 = torch.autograd.grad(selectivity_score.sum(), ligand_v_next_prob,grad_outputs=torch.ones_like(selectivity_score.sum()),
                                       retain_graph=True, allow_unused=True)[0]
            # Position gradient: w.r.t log(selectivity_score) for gradient scaling
            grad2 = torch.autograd.grad(selectivity_score_log.sum(), ligand_pos,grad_outputs=torch.ones_like(selectivity_score.sum()),
                                       retain_graph=False, allow_unused=True)[0]
                                       


            # Handle missing gradients
            if grad1 is None:
                grad1 = torch.zeros_like(ligand_v_next_prob)
            if grad2 is None:
                grad2 = torch.zeros_like(ligand_pos)

            return grad1, grad2, selectivity_score, v_on_pred, v_off_mean

    def joint_sequential_selectivity_guide(self, ligand_v_index, ligand_pos, protein_v, protein_pos,
                                          batch_protein, batch_ligand, off_target_data, t,
                                          w_on=1.0, w_off=1.0):
        """
        Joint model-based TRUE sequential selectivity guidance
        - Step 1: On-target + ligand forward pass → get final_ligand_h
        - Step 2: Each off-target + ligand (using same final_ligand_h) → get v_off_pred
        - True sequential processing without information mixing
        """

        with torch.enable_grad():
            ligand_v_next = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)

            # Step 1: Process on-target + ligand to get reference ligand embedding
            time_step_on = torch.zeros(1, dtype=torch.long, device=protein_pos.device)
            protein_id_on = torch.zeros_like(batch_protein)

            preds_on = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                protein_id=protein_id_on,
                time_step=time_step_on
            )

            # Extract on-target predictions and ligand embedding
            v_on_pred = preds_on.get('v_on_pred', preds_on.get('final_exp_pred'))
            final_ligand_h = preds_on['final_ligand_h']  # Fixed ligand embedding from on-target

            # Step 2: Process each off-target sequentially with same ligand embedding
            v_off_pred_list = []

            # Handle case when off_target_data is empty (on-target only mode)
            if not off_target_data:
                # For on-target only guidance: set v_off_mean = 0
                # This makes selectivity_score = w_on * v_on_pred - w_off * 0 = w_on * v_on_pred
                v_off_pred_list = []  # Empty list to indicate no off-targets
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)
                    off_batch_protein = off_data.get('protein_element_batch',
                                                   torch.zeros_like(off_protein_pos[:, 0], dtype=torch.long)).to(protein_pos.device)

                # Create single protein_id for this off-target
                off_protein_id = torch.zeros_like(off_batch_protein)
                time_step_off = torch.zeros(1, dtype=torch.long, device=protein_pos.device)

                # Forward pass for this off-target + ligand
                preds_off = self(
                    protein_pos=off_protein_pos,
                    protein_v=off_protein_v,
                    batch_protein=off_batch_protein,
                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v_next,
                    batch_ligand=batch_ligand,
                    protein_id=off_protein_id,
                    time_step=time_step_off
                )

                # Extract off-target protein features and compute binding affinity
                # Use fixed final_ligand_h from on-target, not the new ligand embedding
                off_final_h = preds_off['final_h']
                off_mask_protein = preds_off['mask_ligand'] == False  # Protein mask
                off_protein_h = off_final_h[off_mask_protein]

                # Aggregate off-target protein features
                h_mol_off_target = off_protein_h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

                # Use fixed ligand embedding from on-target (properly aggregated by batch)
                from torch_scatter import scatter_mean
                h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]
                # Ensure matching batch dimensions with off-target protein
                # FIX: Expand off-target protein to match number of ligand samples (not the other way around!)
                if h_mol_ligand.size(0) != h_mol_off_target.size(0):
                    h_mol_off_target = h_mol_off_target[0:1].expand(h_mol_ligand.size(0), -1)

                # Predict binding affinity for this off-target
                # Check expert_pred input dimension and adapt accordingly
                if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
                    # Original architecture: only ligand features
                    expert_input_off = h_mol_ligand
                else:
                    # Current architecture: ligand + protein features
                    expert_input_off = torch.cat([h_mol_ligand, h_mol_off_target], dim=1)
                v_off_pred = self.expert_pred(expert_input_off)
                v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)  # [batch_size, num_off_targets]
                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)  # [batch_size, 1]
            else:
                v_off_pred = torch.empty(v_on_pred.shape[0], 0, device=v_on_pred.device)
                v_off_mean = torch.zeros_like(v_on_pred)

            # Compute selectivity score
            selectivity_score = w_on * v_on_pred - w_off * v_off_mean

            # Apply log for position gradient (like original KGDiff pv_joint_guide)
            # Type gradient: direct selectivity_score
            # Position gradient: log(selectivity_score) for better gradient scaling
            selectivity_score_log = selectivity_score.log()

            # Compute gradients with respect to selectivity score
            # Type gradient: w.r.t selectivity_score (not log)
            grad1 = torch.autograd.grad(selectivity_score, ligand_v_next,grad_outputs=torch.ones_like(selectivity_score),
                                       retain_graph=True, allow_unused=True)[0]
            # Position gradient: w.r.t log(selectivity_score) for gradient scaling
            grad2 = torch.autograd.grad(selectivity_score_log, ligand_pos,grad_outputs=torch.ones_like(selectivity_score),
                                       retain_graph=False, allow_unused=True)[0]

            # Handle missing gradients
            if grad1 is None:
                grad1 = torch.zeros_like(ligand_v_next)
            if grad2 is None:
                grad2 = torch.zeros_like(ligand_pos)

            return grad1, grad2, selectivity_score, v_on_pred, v_off_mean

    def joint_sequential_selectivity_guide_no_interaction(self, ligand_v_index, ligand_pos, protein_v, protein_pos,
                                                          batch_protein, batch_ligand, off_target_data, t,
                                                          w_on=1.0, w_off=1.0):
        """
        NEW: Joint model-based sequential selectivity guidance with NO off-target interaction
        - Step 1: On-target + ligand forward pass WITH INTERACTION → get final_ligand_h and v_on_pred
        - Step 2: Each off-target protein ONLY (NO LIGAND) → get off_protein_h
        - Step 3: Combine on-target ligand embedding + off-target protein embedding → predict v_off_pred
        - NO protein-ligand interaction for off-target (only atom embeddings + internal protein distances)

        Key difference from joint_sequential_selectivity_guide():
        - Off-target protein processed WITHOUT any ligand involvement
        - No protein-ligand edges, no cross-attention, no 3D interaction between off-target and ligand
        """

        with torch.enable_grad():
            ligand_v_next = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)

            # Step 1: Process on-target + ligand WITH INTERACTION
            time_step_on = torch.zeros(1, dtype=torch.long, device=protein_pos.device)
            protein_id_on = torch.zeros_like(batch_protein)

            preds_on = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                protein_id=protein_id_on,
                time_step=time_step_on
            )

            # Extract on-target predictions and ligand embedding (WITH interaction)
            v_on_pred = preds_on.get('v_on_pred', preds_on.get('final_exp_pred'))
            final_ligand_h = preds_on['final_ligand_h']  # Ligand embedding from on-target WITH interaction

            # Step 2: Process each off-target protein SEPARATELY WITHOUT LIGAND
            v_off_pred_list = []

            # Handle case when off_target_data is empty (on-target only mode)
            if not off_target_data:
                v_off_pred_list = []  # Empty list to indicate no off-targets
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)
                    off_batch_protein = off_data.get('protein_element_batch',
                                                   torch.zeros_like(off_protein_pos[:, 0], dtype=torch.long)).to(protein_pos.device)

                    # === Process off-target protein ONLY (NO LIGAND, NO INTERACTION) ===
                    # Embed off-target protein atoms
                    h_off_protein = self.protein_atom_emb(off_protein_v)

                    # Add node indicator if enabled
                    if self.config.node_indicator:
                        h_off_protein = torch.cat([h_off_protein, torch.zeros(len(h_off_protein), 1).to(h_off_protein)], -1)

                    # Create mask: all nodes are protein (NO ligand)
                    mask_protein_only = torch.zeros(len(h_off_protein), dtype=torch.bool, device=h_off_protein.device)

                    # Process through RefineNet WITHOUT ligand
                    # Only protein-protein edges will be created (NO protein-ligand edges)
                    outputs_off = self.refine_net(
                        h_off_protein,
                        off_protein_pos,
                        mask_protein_only,  # All False = all protein nodes
                        off_batch_protein,
                        protein_id=None,
                        return_all=False,
                        fix_x=False
                    )

                    # Aggregate off-target protein features (NO ligand interaction involved)
                    from torch_scatter import scatter_mean
                    h_mol_off_protein = scatter_mean(outputs_off['h'], off_batch_protein, dim=0)  # [1, hidden_dim]

                    # Use fixed ligand embedding from on-target (WITH interaction)
                    h_mol_ligand = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # [batch_size, hidden_dim]

                    # Ensure matching batch dimensions
                    # FIX: Expand off-target protein to match number of ligand samples (not the other way around!)
                    if h_mol_ligand.size(0) != h_mol_off_protein.size(0):
                        h_mol_off_protein = h_mol_off_protein[0:1].expand(h_mol_ligand.size(0), -1)

                    # Predict binding affinity: on-target ligand + off-target protein (NO interaction)
                    if hasattr(self, 'expert_input_dim') and self.expert_input_dim == self.hidden_dim:
                        # Original architecture: only ligand features
                        expert_input_off = h_mol_ligand
                    else:
                        # Current architecture: ligand + protein features
                        expert_input_off = torch.cat([h_mol_ligand, h_mol_off_protein], dim=1)

                    v_off_pred = self.expert_pred(expert_input_off)
                    v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)  # [batch_size, num_off_targets]
                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)  # [batch_size, 1]
            else:
                v_off_pred = torch.empty(v_on_pred.shape[0], 0, device=v_on_pred.device)
                v_off_mean = torch.zeros_like(v_on_pred)

            # Compute selectivity score
            selectivity_score = w_on * v_on_pred - w_off * v_off_mean

            # Apply log for position gradient (like original KGDiff pv_joint_guide)
            # Type gradient: direct selectivity_score
            # Position gradient: log(selectivity_score) for better gradient scaling
            selectivity_score_log = selectivity_score.log()

            # Compute gradients with respect to selectivity score
            # Type gradient: w.r.t selectivity_score (not log)
            grad1 = torch.autograd.grad(selectivity_score, ligand_v_next, grad_outputs=torch.ones_like(selectivity_score),
                                       retain_graph=True, allow_unused=True)[0]
            # Position gradient: w.r.t log(selectivity_score) for gradient scaling
            grad2 = torch.autograd.grad(selectivity_score_log, ligand_pos, grad_outputs=torch.ones_like(selectivity_score),
                                       retain_graph=False, allow_unused=True)[0]

            # Handle missing gradients
            if grad1 is None:
                grad1 = torch.zeros_like(ligand_v_next)
            if grad2 is None:
                grad2 = torch.zeros_like(ligand_pos)

            return grad1, grad2, selectivity_score, v_on_pred, v_off_mean

    def joint_on_off_no_interaction_sequential(self, ligand_v_index, ligand_pos, protein_v, protein_pos,
                                                batch_protein, batch_ligand, off_target_data, t,
                                                w_on=1.0, w_off=1.0):
        """
        NEW SEQUENTIAL STRATEGY: Apply on-target and off-target gradients SEQUENTIALLY within single timestep

        Key difference from joint_sequential_selectivity_guide_no_interaction():
        - That function: combines gradients into selectivity_score = w_on*V_on - w_off*V_off
          → Problem: gradient cancellation when both gradients aligned

        - This function: Sequential application
          → Step 1: Compute on-target gradient (head1 with interaction) and apply to get refined state
          → Step 2: Compute off-target gradient (head2 without interaction) ON REFINED STATE (not original state)
          → No gradient cancellation, on-target improvement guaranteed

        Head1 (on-target): Uses complex with interaction (original KGDiff)
        Head2 (off-target): Uses protein + ligand embeddings without interaction (concat, attention, or cross-attention)

        Returns: on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_pred
        """

        # Determine which head to use for no-interaction prediction (head2)
        # Print info only once using a class attribute flag
        if not hasattr(self, '_joint_no_interaction_initialized'):
            if hasattr(self, 'cross_attn_affinity_head'):
                # Atom-level cross-attention model: use cross_attn_affinity_head
                expert_predictor_off = self.cross_attn_affinity_head
                self.use_cross_attn_guidance_off = True
                print("[INFO] joint_on_off_no_interaction_sequential: Using cross_attn_affinity_head (Atom-level Cross-Attention Head2) for off-target")
            elif hasattr(self, 'non_interaction_affinity_head'):
                # Sam-PL dual-head model: use head2 (non-interaction head)
                expert_predictor_off = self.non_interaction_affinity_head
                self.use_cross_attn_guidance_off = False
                print("[INFO] joint_on_off_no_interaction_sequential: Using non_interaction_affinity_head (Sam-PL Head2) for off-target")
            elif hasattr(self, 'expert_pred_off'):
                # Dual-head BA model: use expert_pred_off
                expert_predictor_off = self.expert_pred_off
                self.use_cross_attn_guidance_off = False
                print("[INFO] joint_on_off_no_interaction_sequential: Using expert_pred_off (dedicated no-interaction head) for off-target")
            else:
                # Fallback to expert_pred
                expert_predictor_off = self.expert_pred
                self.use_cross_attn_guidance_off = False
                print("[INFO] joint_on_off_no_interaction_sequential: Using expert_pred (fallback) for off-target")
            self._joint_no_interaction_initialized = True
        else:
            # Already initialized, just get the predictor without printing
            if hasattr(self, 'cross_attn_affinity_head'):
                expert_predictor_off = self.cross_attn_affinity_head
            elif hasattr(self, 'non_interaction_affinity_head'):
                expert_predictor_off = self.non_interaction_affinity_head
            elif hasattr(self, 'expert_pred_off'):
                expert_predictor_off = self.expert_pred_off
            else:
                expert_predictor_off = self.expert_pred

        with torch.enable_grad():
            ligand_v_next = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            ligand_pos_original = ligand_pos.detach().requires_grad_(True)

            # ===== STEP 1: On-target Guidance (PRIMARY) - Head1 with interaction =====
            # FIX: Use actual timestep t instead of always 0
            time_step_on = t.to(protein_pos.device)  # Use the actual timestep from diffusion
            protein_id_on = torch.zeros_like(batch_protein)

            # Forward pass with on-target protein (WITH INTERACTION - original KGDiff)
            preds_on = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos_original,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                protein_id=protein_id_on,
                time_step=time_step_on
            )

            v_on_pred = preds_on.get('v_on_pred', preds_on.get('final_exp_pred'))
            final_ligand_h = preds_on['final_ligand_h']  # Save for off-target prediction

            # FIX: Ensure v_on_pred is at least 1D (not scalar)
            if v_on_pred.dim() == 0:
                v_on_pred = v_on_pred.unsqueeze(0)

            # Compute on-target gradients (maximize on-target affinity)
            v_on_pred_log = v_on_pred.log()

            # Type gradient: w.r.t v_on_pred (not log)
            on_grad_v = torch.autograd.grad(v_on_pred, ligand_v_next,
                                           grad_outputs=torch.ones_like(v_on_pred),
                                           retain_graph=True, allow_unused=True)[0]
            # Position gradient: w.r.t log(v_on_pred) for gradient scaling
            # FIX: Keep retain_graph=True so we can compute off-target gradients later
            on_grad_pos = torch.autograd.grad(v_on_pred_log, ligand_pos_original,
                                             grad_outputs=torch.ones_like(v_on_pred),
                                             retain_graph=True, allow_unused=True)[0]

            # Handle missing gradients
            if on_grad_v is None:
                on_grad_v = torch.zeros_like(ligand_v_next)
            if on_grad_pos is None:
                on_grad_pos = torch.zeros_like(ligand_pos_original)

        # ===== STEP 2: Off-target Guidance (SECONDARY) - Head2 without interaction =====
        # Process ligand independently (NO protein interaction) for Head2
        with torch.enable_grad():
            # Import scatter_mean at the beginning
            from torch_scatter import scatter_mean

            # First, process ligand independently (NO protein) to get ligand embedding without interaction
            h_ligand = self.ligand_atom_emb(ligand_v_next)

            # Add node indicator if enabled
            if self.config.node_indicator:
                h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1).to(h_ligand)], -1)

            # Create mask: all nodes are ligand (NO protein)
            mask_ligand_only = torch.ones(len(h_ligand), dtype=torch.bool, device=h_ligand.device)

            # Process through RefineNet WITHOUT protein
            outputs_ligand = self.refine_net(
                h_ligand,
                ligand_pos_original,
                mask_ligand_only,
                batch_ligand,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            # Aggregate ligand features (NO interaction)
            final_ligand_h_no_interaction = scatter_mean(outputs_ligand['h'], batch_ligand, dim=0)

            # Process each off-target protein SEPARATELY WITHOUT LIGAND
            v_off_pred_list = []

            # DEBUG: Check off_target_data
            if not off_target_data:
                v_off_pred_list = []
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)

                    # Get batch indices for off-target protein (ensure 1D tensor)
                    if hasattr(off_data, 'protein_element_batch'):
                        off_batch_protein = off_data.protein_element_batch.to(protein_pos.device)
                    else:
                        # Create default batch indices (all atoms belong to batch 0)
                        off_batch_protein = torch.zeros(off_protein_pos.shape[0], dtype=torch.long, device=protein_pos.device)

                    # Ensure it's 1D
                    if off_batch_protein.dim() > 1:
                        off_batch_protein = off_batch_protein.squeeze()

                    # Embed off-target protein atoms
                    h_off_protein = self.protein_atom_emb(off_protein_v)

                    # Add node indicator if enabled
                    if self.config.node_indicator:
                        h_off_protein = torch.cat([h_off_protein, torch.zeros(len(h_off_protein), 1).to(h_off_protein)], -1)

                    # Create mask: all nodes are protein (NO ligand)
                    mask_protein_only = torch.zeros(len(h_off_protein), dtype=torch.bool, device=h_off_protein.device)

                    # Process through RefineNet WITHOUT ligand
                    outputs_off = self.refine_net(
                        h_off_protein,
                        off_protein_pos,
                        mask_protein_only,
                        off_batch_protein,
                        protein_id=None,
                        return_all=False,
                        fix_x=False
                    )

                    # Aggregate off-target protein features
                    final_off_protein_h = scatter_mean(outputs_off['h'], off_batch_protein, dim=0)

                    # Ensure matching batch dimensions
                    # FIX: Expand off-target protein to match number of ligand samples (not the other way around!)
                    if final_ligand_h_no_interaction.size(0) != final_off_protein_h.size(0):
                        final_off_protein_h = final_off_protein_h[0:1].expand(final_ligand_h_no_interaction.size(0), -1)

                    # ===== Head2 Prediction: Use appropriate architecture =====
                    # Check which variant of Head 2 we're using
                    if hasattr(self, 'cross_attn_query'):
                        # VARIANT 1: Atom-level cross-attention Head 2
                        # Use atom-level features from outputs_ligand and outputs_off
                        protein_h_atom_off = outputs_off['h']  # [num_protein_atoms, hidden_dim]
                        ligand_h_atom_off = outputs_ligand['h']  # [num_ligand_atoms, hidden_dim]

                        # Project to Q, K, V
                        Q_off = self.cross_attn_query(protein_h_atom_off)
                        K_off = self.cross_attn_key(ligand_h_atom_off)
                        V_off = self.cross_attn_value(ligand_h_atom_off)

                        # Reshape for multi-head attention
                        Q_off = Q_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                        K_off = K_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                        V_off = V_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)

                        # IMPORTANT: Compute attention per batch to avoid cross-contamination
                        # FIX: Loop over ligand samples, not protein batches (protein might be single structure)
                        batch_size_max_off = batch_ligand.max().item() + 1
                        attended_protein_h_off_list = []

                        for b in range(batch_size_max_off):
                            # For multi-protein case: protein_mask_b selects off-protein for sample b
                            # For single-protein case: protein_mask_b selects all off-protein atoms
                            protein_mask_b = off_batch_protein == b if off_batch_protein.max().item() >= b else torch.ones_like(off_batch_protein, dtype=torch.bool)
                            ligand_mask_b = batch_ligand == b

                            Q_off_b = Q_off[protein_mask_b]  # [num_protein_atoms_b, num_heads, head_dim]
                            K_off_b = K_off[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]
                            V_off_b = V_off[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]

                            # Compute attention: Q @ K^T / sqrt(d_k)
                            attn_scores_off_b = torch.einsum('phd,lhd->phl', Q_off_b, K_off_b) / np.sqrt(self.cross_attn_head_dim)
                            attn_weights_off_b = torch.softmax(attn_scores_off_b, dim=-1)

                            # Apply attention to values: attn @ V
                            attended_off_b = torch.einsum('phl,lhd->phd', attn_weights_off_b, V_off_b)
                            attended_off_b = attended_off_b.reshape(-1, self.hidden_dim)

                            # Aggregate this sample's attended features immediately
                            attended_off_b_mean = attended_off_b.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                            attended_protein_h_off_list.append(attended_off_b_mean)

                        # Stack all samples to get [num_samples, hidden_dim]
                        h_mol_attended_off = torch.cat(attended_protein_h_off_list, dim=0)  # [num_ligand_samples, hidden_dim]

                        # Output projection
                        h_mol_attended_off = self.cross_attn_output(h_mol_attended_off)  # [num_ligand_samples, hidden_dim]

                        # Predict affinity from attended features
                        v_off_pred = expert_predictor_off(h_mol_attended_off)

                    elif hasattr(self, 'non_interaction_query'):
                        # VARIANT 2: Attention-based Head 2 (molecule-level)
                        # Query: protein embedding, Key/Value: ligand embedding
                        query_off = self.non_interaction_query(final_off_protein_h)  # [batch_size, hidden_dim]
                        key_off = self.non_interaction_key(final_ligand_h_no_interaction)  # [batch_size, hidden_dim]
                        value_off = self.non_interaction_value(final_ligand_h_no_interaction)  # [batch_size, hidden_dim]

                        # Compute attention scores: Q * K^T / sqrt(d_k)
                        attention_scores_off = torch.sum(query_off * key_off, dim=-1, keepdim=True) / self.non_interaction_scale  # [batch_size, 1]

                        # For single query-key pairs (batch_size=1), use sigmoid instead of softmax
                        if final_off_protein_h.size(0) == 1:
                            attention_weights_off = torch.sigmoid(attention_scores_off)
                        else:
                            attention_weights_off = torch.softmax(attention_scores_off, dim=0)

                        # Combine protein (query) and ligand (value) information
                        attended_features_off = attention_weights_off * value_off + (1 - attention_weights_off) * query_off  # [batch_size, hidden_dim]

                        # Predict affinity from attended features
                        v_off_pred = expert_predictor_off(attended_features_off)

                    else:
                        # VARIANT 3: Concatenation-based Head 2 (original simple implementation)
                        # Combine ligand + off-target protein (both NO interaction)
                        expert_input_off = torch.cat([final_ligand_h_no_interaction, final_off_protein_h], dim=1)

                        # Use expert predictor (head2 for no-interaction prediction)
                        v_off_pred = expert_predictor_off(expert_input_off)

                    v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)
                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)
            else:
                v_off_pred = torch.empty(v_on_pred.shape[0], 0, device=v_on_pred.device)
                v_off_mean = torch.zeros_like(v_on_pred)

            # FIX: Ensure v_off_mean is at least 1D (not scalar)
            if v_off_mean.dim() == 0:
                v_off_mean = v_off_mean.unsqueeze(0)

            # Compute off-target gradients (minimize off-target affinity)
            # FIX: Compute gradients w.r.t ORIGINAL ligand variables (ligand_v_next, ligand_pos_original)
            # NOT the detached copies, so gradient can flow through final_ligand_h
            if v_off_mean.sum() > 0:  # Only compute gradient if there are off-targets
                v_off_mean_log = v_off_mean.log()

                # Type gradient: w.r.t v_off_mean (not log)
                # FIX: Use ligand_v_next (original variable with gradient)
                off_grad_v = torch.autograd.grad(v_off_mean, ligand_v_next,
                                                grad_outputs=torch.ones_like(v_off_mean),
                                                retain_graph=True, allow_unused=True)[0]
                # Position gradient: w.r.t log(v_off_mean) for gradient scaling
                # FIX: Use ligand_pos_original (original variable with gradient)
                off_grad_pos = torch.autograd.grad(v_off_mean_log, ligand_pos_original,
                                                  grad_outputs=torch.ones_like(v_off_mean),
                                                  retain_graph=False, allow_unused=True)[0]

                if off_grad_v is None:
                    off_grad_v = torch.zeros_like(ligand_v_next)
                    print("[WARNING] Off-target type gradient is None - no gradient flow!")
                if off_grad_pos is None:
                    off_grad_pos = torch.zeros_like(ligand_pos_original)
                    print("[WARNING] Off-target position gradient is None - no gradient flow!")
            else:
                off_grad_v = torch.zeros_like(ligand_v_next)
                off_grad_pos = torch.zeros_like(ligand_pos_original)

        # Return SEPARATE gradients (not combined)
        # Caller will apply:
        #   1. on_grad first to get refined state
        #   2. off_grad to refined state
        return on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_mean

    def head1_only_sequential_guidance(self, ligand_v_index, ligand_pos, protein_v, protein_pos,
                                       batch_protein, batch_ligand, off_target_data, t,
                                       w_on=1.0, w_off=1.0):
        """
        NEW HEAD1-ONLY SEQUENTIAL GUIDANCE: Use interaction-based head1 for both on-target and off-target

        Strategy:
        1. Process on-target protein-ligand complex (WITH interaction) -> final_on_complex_h
        2. Process off-target protein-ligand complexes (WITH interaction) -> final_off_complex_h
        3. Predict using head1 (interaction_affinity_head):
           - v_on = head1(ligand_h_from_on_complex + protein_h_from_on_complex)
           - v_off = head1(ligand_h_from_off_complex + protein_h_from_off_complex)
        4. Apply sequential gradients like joint_on_off_no_interaction_sequential

        Returns: on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_pred
        """

        # Determine which head to use for interaction-based prediction
        if not hasattr(self, '_head1_guidance_initialized'):
            if hasattr(self, 'interaction_affinity_head'):
                # Sam-PL dual-head model: use head1 (interaction head)
                expert_predictor = self.interaction_affinity_head
                print("[INFO] Using interaction_affinity_head (Sam-PL Head1)")
            elif hasattr(self, 'expert_pred_on'):
                # Dual-head BA model: use expert_pred_on
                expert_predictor = self.expert_pred_on
                print("[INFO] Using expert_pred_on (dedicated interaction head)")
            elif hasattr(self, 'expert_pred'):
                # Fallback to expert_pred
                expert_predictor = self.expert_pred
                print("[INFO] Using expert_pred (fallback)")
            else:
                raise ValueError("No suitable affinity prediction head found")
            self._head1_guidance_initialized = True
        else:
            # Already initialized, just get the predictor without printing
            if hasattr(self, 'interaction_affinity_head'):
                expert_predictor = self.interaction_affinity_head
            elif hasattr(self, 'expert_pred_on'):
                expert_predictor = self.expert_pred_on
            else:
                expert_predictor = self.expert_pred

        with torch.enable_grad():
            ligand_v_next = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            ligand_pos_original = ligand_pos.detach().requires_grad_(True)

            # ===== STEP 1: Process On-target Complex (WITH interaction) =====
            # Embed ligand
            h_ligand = self.ligand_atom_emb(ligand_v_next)

            # Embed on-target protein
            h_on_protein = self.protein_atom_emb(protein_v)

            # Add node indicators if enabled
            if self.config.node_indicator:
                h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1).to(h_ligand)], -1)
                h_on_protein = torch.cat([h_on_protein, torch.zeros(len(h_on_protein), 1).to(h_on_protein)], -1)

            # Combine ligand and protein into single graph (WITH interaction)
            h_combined = torch.cat([h_ligand, h_on_protein], dim=0)
            pos_combined = torch.cat([ligand_pos_original, protein_pos], dim=0)

            # Create mask: ligand=True, protein=False
            mask_combined = torch.cat([
                torch.ones(len(h_ligand), dtype=torch.bool, device=h_ligand.device),
                torch.zeros(len(h_on_protein), dtype=torch.bool, device=h_on_protein.device)
            ])

            # Create batch indices
            batch_combined = torch.cat([batch_ligand, batch_protein], dim=0)

            # Process through RefineNet WITH interaction
            outputs_on_complex = self.refine_net(
                h_combined,
                pos_combined,
                mask_combined,
                batch_combined,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            # Separate ligand and protein embeddings
            num_ligand_atoms = len(h_ligand)
            h_ligand_from_on_complex = outputs_on_complex['h'][:num_ligand_atoms]
            h_protein_from_on_complex = outputs_on_complex['h'][num_ligand_atoms:]

            # Aggregate to per-graph level
            final_ligand_h_on = scatter_mean(h_ligand_from_on_complex, batch_ligand, dim=0)
            final_protein_h_on = scatter_mean(h_protein_from_on_complex, batch_protein, dim=0)

            # ===== STEP 2: Predict On-target Binding Affinity using Head1 =====
            # Combine ligand + on-target protein embeddings
            expert_input_on = torch.cat([final_ligand_h_on, final_protein_h_on], dim=1)

            # Use expert predictor (head1 for interaction-based prediction)
            v_on_pred = expert_predictor(expert_input_on)

            # Compute on-target gradients
            v_on_pred_log = v_on_pred.log()

            # Type gradient: w.r.t v_on_pred (not log)
            on_grad_v = torch.autograd.grad(v_on_pred, ligand_v_next,
                                           grad_outputs=torch.ones_like(v_on_pred),
                                           retain_graph=True, allow_unused=True)[0]
            # Position gradient: w.r.t log(v_on_pred) for gradient scaling
            on_grad_pos = torch.autograd.grad(v_on_pred_log, ligand_pos_original,
                                             grad_outputs=torch.ones_like(v_on_pred),
                                             retain_graph=True, allow_unused=True)[0]

            # Handle missing gradients
            if on_grad_v is None:
                on_grad_v = torch.zeros_like(ligand_v_next)
            if on_grad_pos is None:
                on_grad_pos = torch.zeros_like(ligand_pos_original)

            # ===== STEP 3: Process Off-target Complexes and Predict =====
            v_off_pred_list = []

            if not off_target_data:
                v_off_pred_list = []
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)

                    # Get batch indices for off-target protein
                    if hasattr(off_data, 'protein_element_batch'):
                        off_batch_protein = off_data.protein_element_batch.to(protein_pos.device)
                    else:
                        off_batch_protein = torch.zeros(off_protein_pos.shape[0], dtype=torch.long, device=protein_pos.device)

                    # Ensure it's 1D
                    if off_batch_protein.dim() > 1:
                        off_batch_protein = off_batch_protein.squeeze()

                    # Embed off-target protein
                    h_off_protein = self.protein_atom_emb(off_protein_v)

                    # Add node indicators if enabled
                    if self.config.node_indicator:
                        h_off_protein = torch.cat([h_off_protein, torch.zeros(len(h_off_protein), 1).to(h_off_protein)], -1)

                    # Combine ligand and off-target protein into single graph (WITH interaction)
                    h_combined_off = torch.cat([h_ligand, h_off_protein], dim=0)
                    pos_combined_off = torch.cat([ligand_pos_original, off_protein_pos], dim=0)

                    # Create mask: ligand=True, protein=False
                    mask_combined_off = torch.cat([
                        torch.ones(len(h_ligand), dtype=torch.bool, device=h_ligand.device),
                        torch.zeros(len(h_off_protein), dtype=torch.bool, device=h_off_protein.device)
                    ])

                    # Create batch indices (match batch sizes)
                    batch_combined_off = torch.cat([batch_ligand, off_batch_protein], dim=0)

                    # Process through RefineNet WITH interaction
                    outputs_off_complex = self.refine_net(
                        h_combined_off,
                        pos_combined_off,
                        mask_combined_off,
                        batch_combined_off,
                        protein_id=None,
                        return_all=False,
                        fix_x=False
                    )

                    # Separate ligand and protein embeddings
                    h_ligand_from_off_complex = outputs_off_complex['h'][:num_ligand_atoms]
                    h_protein_from_off_complex = outputs_off_complex['h'][num_ligand_atoms:]

                    # Aggregate to per-graph level
                    final_ligand_h_off = scatter_mean(h_ligand_from_off_complex, batch_ligand, dim=0)
                    final_protein_h_off = scatter_mean(h_protein_from_off_complex, off_batch_protein, dim=0)

                    # Ensure matching batch dimensions
                    # FIX: Expand off-target protein to match ligand samples (not the other way around!)
                    if final_ligand_h_off.size(0) != final_protein_h_off.size(0):
                        final_protein_h_off = final_protein_h_off[0:1].expand(final_ligand_h_off.size(0), -1)
                    final_ligand_h_off_matched = final_ligand_h_off

                    # Combine ligand + off-target protein (both WITH interaction)
                    expert_input_off = torch.cat([final_ligand_h_off_matched, final_protein_h_off], dim=1)

                    # Use expert predictor (head1 for interaction-based prediction)
                    v_off_pred = expert_predictor(expert_input_off)
                    v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)
                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)
            else:
                v_off_pred = torch.empty(v_on_pred.shape[0], 0, device=v_on_pred.device)
                v_off_mean = torch.zeros_like(v_on_pred)

            # Compute off-target gradients (minimize off-target affinity)
            if v_off_mean.sum() > 0:  # Only compute gradient if there are off-targets
                v_off_mean_log = v_off_mean.log()

                # Type gradient: w.r.t v_off_mean (not log)
                off_grad_v = torch.autograd.grad(v_off_mean, ligand_v_next,
                                                grad_outputs=torch.ones_like(v_off_mean),
                                                retain_graph=True, allow_unused=True)[0]
                # Position gradient: w.r.t log(v_off_mean) for gradient scaling
                off_grad_pos = torch.autograd.grad(v_off_mean_log, ligand_pos_original,
                                                  grad_outputs=torch.ones_like(v_off_mean),
                                                  retain_graph=False, allow_unused=True)[0]

                if off_grad_v is None:
                    off_grad_v = torch.zeros_like(ligand_v_next)
                    print("[WARNING] Off-target type gradient is None - no gradient flow!")
                if off_grad_pos is None:
                    off_grad_pos = torch.zeros_like(ligand_pos_original)
                    print("[WARNING] Off-target position gradient is None - no gradient flow!")
            else:
                off_grad_v = torch.zeros_like(ligand_v_next)
                off_grad_pos = torch.zeros_like(ligand_pos_original)

        # Return SEPARATE gradients (not combined)
        return on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_mean

    def head2_only_sequential_guidance(self, ligand_v_index, ligand_pos, protein_v, protein_pos,
                                       batch_protein, batch_ligand, off_target_data, t,
                                       w_on=1.0, w_off=1.0):
        """
        NEW HEAD2-ONLY SEQUENTIAL GUIDANCE: Use interaction-free head2 for both on-target and off-target

        Strategy:
        1. Process on-target protein independently (NO ligand interaction) -> final_on_protein_h
        2. Process ligand independently (NO protein interaction) -> final_ligand_h_no_interaction
        3. Process off-target proteins independently (NO ligand interaction) -> final_off_protein_h
        4. Predict using head2 (expert_pred_off):
           - v_on = head2(final_ligand_h_no_interaction + final_on_protein_h)
           - v_off = head2(final_ligand_h_no_interaction + final_off_protein_h)
        5. Apply sequential gradients like joint_on_off_no_interaction_sequential

        Returns: on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_pred
        """

        # Determine which head to use for no-interaction prediction
        # Print info only once using a class attribute flag
        if not hasattr(self, '_head2_guidance_initialized'):
            if hasattr(self, 'cross_attn_affinity_head'):
                # Atom-level cross-attention model: use cross_attn_affinity_head
                expert_predictor = self.cross_attn_affinity_head
                self.use_cross_attn_guidance = True
                print("[INFO] Using cross_attn_affinity_head (Atom-level Cross-Attention Head2)")
            elif hasattr(self, 'non_interaction_affinity_head'):
                # Sam-PL dual-head model: use head2 (non-interaction head)
                expert_predictor = self.non_interaction_affinity_head
                self.use_cross_attn_guidance = False
                print("[INFO] Using non_interaction_affinity_head (Sam-PL Head2)")
            elif hasattr(self, 'expert_pred_off'):
                # Dual-head BA model: use expert_pred_off
                expert_predictor = self.expert_pred_off
                self.use_cross_attn_guidance = False
                print("[INFO] Using expert_pred_off (dedicated no-interaction head)")
            else:
                # Fallback to expert_pred
                expert_predictor = self.expert_pred
                self.use_cross_attn_guidance = False
                print("[INFO] Using expert_pred (fallback)")
            self._head2_guidance_initialized = True
        else:
            # Already initialized, just get the predictor without printing
            if hasattr(self, 'cross_attn_affinity_head'):
                expert_predictor = self.cross_attn_affinity_head
            elif hasattr(self, 'non_interaction_affinity_head'):
                expert_predictor = self.non_interaction_affinity_head
            elif hasattr(self, 'expert_pred_off'):
                expert_predictor = self.expert_pred_off
            else:
                expert_predictor = self.expert_pred

        with torch.enable_grad():
            ligand_v_next = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            ligand_pos_original = ligand_pos.detach().requires_grad_(True)

            # ===== STEP 1: Process Ligand Independently (NO protein interaction) =====
            # Embed ligand
            h_ligand = self.ligand_atom_emb(ligand_v_next)

            # Add node indicator if enabled
            if self.config.node_indicator:
                h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1).to(h_ligand)], -1)

            # Create mask: all nodes are ligand (NO protein)
            mask_ligand_only = torch.ones(len(h_ligand), dtype=torch.bool, device=h_ligand.device)

            # Process through RefineNet WITHOUT protein
            outputs_ligand = self.refine_net(
                h_ligand,
                ligand_pos_original,
                mask_ligand_only,
                batch_ligand,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            # Aggregate ligand features (NO interaction)
            final_ligand_h_no_interaction = scatter_mean(outputs_ligand['h'], batch_ligand, dim=0)

            # ===== STEP 2: Process On-target Protein Independently (NO ligand interaction) =====
            # Embed on-target protein
            h_on_protein = self.protein_atom_emb(protein_v)

            # Add node indicator if enabled
            if self.config.node_indicator:
                h_on_protein = torch.cat([h_on_protein, torch.zeros(len(h_on_protein), 1).to(h_on_protein)], -1)

            # Create mask: all nodes are protein (NO ligand)
            mask_protein_only = torch.zeros(len(h_on_protein), dtype=torch.bool, device=h_on_protein.device)

            # Process through RefineNet WITHOUT ligand
            outputs_on_protein = self.refine_net(
                h_on_protein,
                protein_pos,
                mask_protein_only,
                batch_protein,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            # Aggregate on-target protein features
            final_on_protein_h = scatter_mean(outputs_on_protein['h'], batch_protein, dim=0)

            # ===== STEP 3: Predict On-target Binding Affinity using Head2 =====
            # Ensure matching batch dimensions
            # FIX: Expand protein to match ligand samples (not the other way around!)
            if final_ligand_h_no_interaction.size(0) != final_on_protein_h.size(0):
                final_on_protein_h = final_on_protein_h[0:1].expand(final_ligand_h_no_interaction.size(0), -1)
            final_ligand_h_no_interaction_expanded = final_ligand_h_no_interaction

            # Check which variant of Head 2 we're using
            if hasattr(self, 'cross_attn_query'):
                # VARIANT 1: Atom-level cross-attention Head 2
                # Use atom-level features from outputs_ligand and outputs_on_protein
                protein_h_atom = outputs_on_protein['h']  # [num_protein_atoms, hidden_dim]
                ligand_h_atom = outputs_ligand['h']  # [num_ligand_atoms, hidden_dim]

                # Project to Q, K, V
                Q = self.cross_attn_query(protein_h_atom)
                K = self.cross_attn_key(ligand_h_atom)
                V = self.cross_attn_value(ligand_h_atom)

                # Reshape for multi-head attention
                Q = Q.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                K = K.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                V = V.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)

                # IMPORTANT: Compute attention per batch to avoid cross-contamination
                # FIX: Loop over ligand samples, not protein batches (protein might be single structure)
                batch_size_max = batch_ligand.max().item() + 1
                attended_protein_h_list = []

                for b in range(batch_size_max):
                    # For multi-protein case: protein_mask_b selects protein for sample b
                    # For single-protein case: protein_mask_b selects all protein atoms
                    protein_mask_b = batch_protein == b if batch_protein.max().item() >= b else torch.ones_like(batch_protein, dtype=torch.bool)
                    ligand_mask_b = batch_ligand == b

                    Q_b = Q[protein_mask_b]  # [num_protein_atoms_b, num_heads, head_dim]
                    K_b = K[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]
                    V_b = V[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]

                    # Compute attention: Q @ K^T / sqrt(d_k)
                    attn_scores_b = torch.einsum('phd,lhd->phl', Q_b, K_b) / np.sqrt(self.cross_attn_head_dim)
                    attn_weights_b = torch.softmax(attn_scores_b, dim=-1)  # [num_protein_atoms_b, num_heads, num_ligand_atoms_b]

                    # Apply attention to values: attn @ V
                    attended_b = torch.einsum('phl,lhd->phd', attn_weights_b, V_b)  # [num_protein_atoms_b, num_heads, head_dim]
                    attended_b = attended_b.reshape(-1, self.hidden_dim)  # [num_protein_atoms_b, hidden_dim]

                    # Aggregate this sample's attended features immediately
                    attended_b_mean = attended_b.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                    attended_protein_h_list.append(attended_b_mean)

                # Stack all samples to get [num_samples, hidden_dim]
                h_mol_attended = torch.cat(attended_protein_h_list, dim=0)  # [num_ligand_samples, hidden_dim]

                # Output projection
                h_mol_attended = self.cross_attn_output(h_mol_attended)  # [num_ligand_samples, hidden_dim]

                # No need to expand - h_mol_attended already has correct shape!

                # Predict affinity from attended features
                v_on_pred = expert_predictor(h_mol_attended)
            elif hasattr(self, 'non_interaction_query'):
                # VARIANT 2: Attention-based Head 2 (molecule-level)
                # Query: protein embedding, Key/Value: ligand embedding
                query = self.non_interaction_query(final_on_protein_h)  # [batch_size, hidden_dim]
                key = self.non_interaction_key(final_ligand_h_no_interaction_expanded)  # [batch_size, hidden_dim]
                value = self.non_interaction_value(final_ligand_h_no_interaction_expanded)  # [batch_size, hidden_dim]

                # Compute attention scores: Q * K^T / sqrt(d_k)
                attention_scores = torch.sum(query * key, dim=-1, keepdim=True) / self.non_interaction_scale  # [batch_size, 1]

                # For single query-key pairs (batch_size=1), use sigmoid instead of softmax
                if final_on_protein_h.size(0) == 1:
                    attention_weights = torch.sigmoid(attention_scores)
                else:
                    attention_weights = torch.softmax(attention_scores, dim=0)

                # Combine protein (query) and ligand (value) information
                attended_features = attention_weights * value + (1 - attention_weights) * query  # [batch_size, hidden_dim]

                # Predict affinity from attended features
                v_on_pred = expert_predictor(attended_features)
            else:
                # VARIANT 3: Concatenation-based Head 2
                # Combine ligand + on-target protein (both NO interaction)
                expert_input_on = torch.cat([final_ligand_h_no_interaction_expanded, final_on_protein_h], dim=1)

                # Use expert predictor (head2 for no-interaction prediction)
                v_on_pred = expert_predictor(expert_input_on)

            # Compute on-target gradients
            v_on_pred_log = v_on_pred.log()

            # Type gradient: w.r.t v_on_pred (not log)
            on_grad_v = torch.autograd.grad(v_on_pred, ligand_v_next,
                                           grad_outputs=torch.ones_like(v_on_pred),
                                           retain_graph=True, allow_unused=True)[0]
            # Position gradient: w.r.t log(v_on_pred) for gradient scaling
            on_grad_pos = torch.autograd.grad(v_on_pred_log, ligand_pos_original,
                                             grad_outputs=torch.ones_like(v_on_pred),
                                             retain_graph=True, allow_unused=True)[0]

            # Handle missing gradients
            if on_grad_v is None:
                on_grad_v = torch.zeros_like(ligand_v_next)
            if on_grad_pos is None:
                on_grad_pos = torch.zeros_like(ligand_pos_original)

            # ===== STEP 4: Process Off-target Proteins and Predict =====
            v_off_pred_list = []

            if not off_target_data:
                v_off_pred_list = []
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)

                    # Get batch indices for off-target protein (ensure 1D tensor)
                    if hasattr(off_data, 'protein_element_batch'):
                        off_batch_protein = off_data.protein_element_batch.to(protein_pos.device)
                    else:
                        # Create default batch indices (all atoms belong to batch 0)
                        off_batch_protein = torch.zeros(off_protein_pos.shape[0], dtype=torch.long, device=protein_pos.device)

                    # Ensure it's 1D
                    if off_batch_protein.dim() > 1:
                        off_batch_protein = off_batch_protein.squeeze()

                    # Embed off-target protein
                    h_off_protein = self.protein_atom_emb(off_protein_v)

                    # Add node indicator if enabled
                    if self.config.node_indicator:
                        h_off_protein = torch.cat([h_off_protein, torch.zeros(len(h_off_protein), 1).to(h_off_protein)], -1)

                    # Create mask: all nodes are protein (NO ligand)
                    mask_off_protein_only = torch.zeros(len(h_off_protein), dtype=torch.bool, device=h_off_protein.device)

                    # Process through RefineNet WITHOUT ligand
                    outputs_off_protein = self.refine_net(
                        h_off_protein,
                        off_protein_pos,
                        mask_off_protein_only,
                        off_batch_protein,
                        protein_id=None,
                        return_all=False,
                        fix_x=False
                    )

                    # Aggregate off-target protein features
                    final_off_protein_h = scatter_mean(outputs_off_protein['h'], off_batch_protein, dim=0)

                    # Ensure matching batch dimensions
                    # FIX: Expand off-target protein to match ligand samples (not the other way around!)
                    if final_ligand_h_no_interaction.size(0) != final_off_protein_h.size(0):
                        final_off_protein_h = final_off_protein_h[0:1].expand(final_ligand_h_no_interaction.size(0), -1)
                    final_ligand_h_no_interaction_off = final_ligand_h_no_interaction

                    # Check which variant of Head 2 we're using
                    if hasattr(self, 'cross_attn_query'):
                        # VARIANT 1: Atom-level cross-attention Head 2
                        # Use atom-level features from outputs_ligand and outputs_off_protein
                        protein_h_atom_off = outputs_off_protein['h']  # [num_protein_atoms, hidden_dim]
                        ligand_h_atom_off = outputs_ligand['h']  # [num_ligand_atoms, hidden_dim]

                        # Project to Q, K, V
                        Q_off = self.cross_attn_query(protein_h_atom_off)
                        K_off = self.cross_attn_key(ligand_h_atom_off)
                        V_off = self.cross_attn_value(ligand_h_atom_off)

                        # Reshape for multi-head attention
                        Q_off = Q_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                        K_off = K_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                        V_off = V_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)

                        # IMPORTANT: Compute attention per batch to avoid cross-contamination
                        # FIX: Loop over ligand samples, not protein batches (protein might be single structure)
                        batch_size_max_off = batch_ligand.max().item() + 1
                        attended_protein_h_off_list = []

                        for b in range(batch_size_max_off):
                            # For multi-protein case: protein_mask_b selects off-protein for sample b
                            # For single-protein case: protein_mask_b selects all off-protein atoms
                            protein_mask_b = off_batch_protein == b if off_batch_protein.max().item() >= b else torch.ones_like(off_batch_protein, dtype=torch.bool)
                            ligand_mask_b = batch_ligand == b

                            Q_off_b = Q_off[protein_mask_b]  # [num_protein_atoms_b, num_heads, head_dim]
                            K_off_b = K_off[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]
                            V_off_b = V_off[ligand_mask_b]    # [num_ligand_atoms_b, num_heads, head_dim]

                            # Compute attention: Q @ K^T / sqrt(d_k)
                            attn_scores_off_b = torch.einsum('phd,lhd->phl', Q_off_b, K_off_b) / np.sqrt(self.cross_attn_head_dim)
                            attn_weights_off_b = torch.softmax(attn_scores_off_b, dim=-1)

                            # Apply attention to values: attn @ V
                            attended_off_b = torch.einsum('phl,lhd->phd', attn_weights_off_b, V_off_b)
                            attended_off_b = attended_off_b.reshape(-1, self.hidden_dim)

                            # Aggregate this sample's attended features immediately
                            attended_off_b_mean = attended_off_b.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                            attended_protein_h_off_list.append(attended_off_b_mean)

                        # Stack all samples to get [num_samples, hidden_dim]
                        h_mol_attended_off = torch.cat(attended_protein_h_off_list, dim=0)  # [num_ligand_samples, hidden_dim]

                        # Output projection
                        h_mol_attended_off = self.cross_attn_output(h_mol_attended_off)  # [num_ligand_samples, hidden_dim]

                        # No need to expand - h_mol_attended_off already has correct shape!

                        # Predict affinity from attended features
                        v_off_pred = expert_predictor(h_mol_attended_off)
                    elif hasattr(self, 'non_interaction_query'):
                        # VARIANT 2: Attention-based Head 2 (molecule-level)
                        # Query: protein embedding, Key/Value: ligand embedding
                        query_off = self.non_interaction_query(final_off_protein_h)  # [batch_size, hidden_dim]
                        key_off = self.non_interaction_key(final_ligand_h_no_interaction_off)  # [batch_size, hidden_dim]
                        value_off = self.non_interaction_value(final_ligand_h_no_interaction_off)  # [batch_size, hidden_dim]

                        # Compute attention scores: Q * K^T / sqrt(d_k)
                        attention_scores_off = torch.sum(query_off * key_off, dim=-1, keepdim=True) / self.non_interaction_scale  # [batch_size, 1]

                        # For single query-key pairs (batch_size=1), use sigmoid instead of softmax
                        if final_off_protein_h.size(0) == 1:
                            attention_weights_off = torch.sigmoid(attention_scores_off)
                        else:
                            attention_weights_off = torch.softmax(attention_scores_off, dim=0)

                        # Combine protein (query) and ligand (value) information
                        attended_features_off = attention_weights_off * value_off + (1 - attention_weights_off) * query_off  # [batch_size, hidden_dim]

                        # Predict affinity from attended features
                        v_off_pred = expert_predictor(attended_features_off)
                    else:
                        # VARIANT 3: Concatenation-based Head 2
                        # Combine ligand + off-target protein (both NO interaction)
                        expert_input_off = torch.cat([final_ligand_h_no_interaction_off, final_off_protein_h], dim=1)

                        # Use expert predictor (head2 for no-interaction prediction)
                        v_off_pred = expert_predictor(expert_input_off)

                    v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)
                v_off_mean = torch.mean(v_off_pred, dim=1, keepdim=True)
            else:
                v_off_pred = torch.empty(v_on_pred.shape[0], 0, device=v_on_pred.device)
                v_off_mean = torch.zeros_like(v_on_pred)

            # Compute off-target gradients (minimize off-target affinity)
            if v_off_mean.sum() > 0:  # Only compute gradient if there are off-targets
                v_off_mean_log = v_off_mean.log()

                # Type gradient: w.r.t v_off_mean (not log)
                off_grad_v = torch.autograd.grad(v_off_mean, ligand_v_next,
                                                grad_outputs=torch.ones_like(v_off_mean),
                                                retain_graph=True, allow_unused=True)[0]
                # Position gradient: w.r.t log(v_off_mean) for gradient scaling
                off_grad_pos = torch.autograd.grad(v_off_mean_log, ligand_pos_original,
                                                  grad_outputs=torch.ones_like(v_off_mean),
                                                  retain_graph=False, allow_unused=True)[0]

                if off_grad_v is None:
                    off_grad_v = torch.zeros_like(ligand_v_next)
                    print("[WARNING] Off-target type gradient is None - no gradient flow!")
                if off_grad_pos is None:
                    off_grad_pos = torch.zeros_like(ligand_pos_original)
                    print("[WARNING] Off-target position gradient is None - no gradient flow!")
            else:
                off_grad_v = torch.zeros_like(ligand_v_next)
                off_grad_pos = torch.zeros_like(ligand_pos_original)

        # Return SEPARATE gradients (not combined)
        return on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_mean

    def head1_head2_sequential_guidance(self, ligand_v_index, ligand_pos, protein_v, protein_pos,
                                       batch_protein, batch_ligand, off_target_data, t,
                                       w_on=1.0, w_off=1.0):
        """
        NEW HEAD1-HEAD2 SEQUENTIAL GUIDANCE:
        Use head1 (interaction-based) for on-target, then head2 (no-interaction) for both on-target and off-target

        Strategy:
        1. Head1 (WITH interaction) predicts on-target BA -> on_grad_v_head1, on_grad_pos_head1
        2. Apply head1 on-target guidance to ligand (pos, grad)
        3. Head2 (NO interaction) predicts on-target BA -> on_grad_v_head2, on_grad_pos_head2
        4. Head2 (NO interaction) predicts off-target BA -> off_grad_v_head2, off_grad_pos_head2
        5. Return all gradients for sequential application

        Returns: on_grad_v_head1, on_grad_pos_head1, on_grad_v_head2, on_grad_pos_head2,
                 off_grad_v_head2, off_grad_pos_head2, v_on_pred_head1, v_on_pred_head2, v_off_pred_head2
        """

        # Determine which heads to use
        if not hasattr(self, '_head1_head2_guidance_initialized'):
            # Head1: Interaction-based predictor
            if hasattr(self, 'interaction_affinity_head'):
                head1_predictor = self.interaction_affinity_head
                print("[INFO] Using interaction_affinity_head as Head1 (Sam-PL)")
            elif hasattr(self, 'expert_pred_on'):
                head1_predictor = self.expert_pred_on
                print("[INFO] Using expert_pred_on as Head1")
            elif hasattr(self, 'expert_pred'):
                head1_predictor = self.expert_pred
                print("[INFO] Using expert_pred as Head1 (fallback)")
            else:
                raise ValueError("No suitable interaction-based head found for Head1")

            # Head2: No-interaction predictor
            if hasattr(self, 'cross_attn_affinity_head'):
                head2_predictor = self.cross_attn_affinity_head
                self.use_cross_attn_guidance_h1h2 = True
                print("[INFO] Using cross_attn_affinity_head as Head2 (Atom-level Cross-Attention)")
            elif hasattr(self, 'non_interaction_affinity_head'):
                head2_predictor = self.non_interaction_affinity_head
                self.use_cross_attn_guidance_h1h2 = False
                print("[INFO] Using non_interaction_affinity_head as Head2 (Sam-PL)")
            elif hasattr(self, 'expert_pred_off'):
                head2_predictor = self.expert_pred_off
                self.use_cross_attn_guidance_h1h2 = False
                print("[INFO] Using expert_pred_off as Head2")
            else:
                head2_predictor = self.expert_pred
                self.use_cross_attn_guidance_h1h2 = False
                print("[INFO] Using expert_pred as Head2 (fallback)")

            self.head1_predictor_h1h2 = head1_predictor
            self.head2_predictor_h1h2 = head2_predictor
            self._head1_head2_guidance_initialized = True
        else:
            head1_predictor = self.head1_predictor_h1h2
            head2_predictor = self.head2_predictor_h1h2

        with torch.enable_grad():
            ligand_v_next = F.one_hot(ligand_v_index, self.num_classes).float().detach().requires_grad_(True)
            ligand_pos_original = ligand_pos.detach().requires_grad_(True)

            # ===== PHASE 1: HEAD1 (WITH INTERACTION) ON-TARGET PREDICTION =====
            # Embed ligand
            h_ligand = self.ligand_atom_emb(ligand_v_next)

            # Embed on-target protein
            h_on_protein = self.protein_atom_emb(protein_v)

            # Add node indicators if enabled
            if self.config.node_indicator:
                h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1).to(h_ligand)], -1)
                h_on_protein = torch.cat([h_on_protein, torch.zeros(len(h_on_protein), 1).to(h_on_protein)], -1)

            # Combine ligand and protein into single graph (WITH interaction)
            h_combined = torch.cat([h_ligand, h_on_protein], dim=0)
            pos_combined = torch.cat([ligand_pos_original, protein_pos], dim=0)

            # Create mask: ligand=True, protein=False
            mask_combined = torch.cat([
                torch.ones(len(h_ligand), dtype=torch.bool, device=h_ligand.device),
                torch.zeros(len(h_on_protein), dtype=torch.bool, device=h_on_protein.device)
            ])

            # Create batch indices
            batch_combined = torch.cat([batch_ligand, batch_protein], dim=0)

            # Process through RefineNet WITH interaction
            outputs_on_complex = self.refine_net(
                h_combined,
                pos_combined,
                mask_combined,
                batch_combined,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            # Separate ligand and protein embeddings
            num_ligand_atoms = len(h_ligand)
            h_ligand_from_on_complex = outputs_on_complex['h'][:num_ligand_atoms]
            h_protein_from_on_complex = outputs_on_complex['h'][num_ligand_atoms:]

            # Aggregate to per-graph level
            final_ligand_h_on_head1 = scatter_mean(h_ligand_from_on_complex, batch_ligand, dim=0)
            final_protein_h_on_head1 = scatter_mean(h_protein_from_on_complex, batch_protein, dim=0)

            # Predict using Head1 (interaction-based)
            expert_input_on_head1 = torch.cat([final_ligand_h_on_head1, final_protein_h_on_head1], dim=1)
            v_on_pred_head1 = head1_predictor(expert_input_on_head1)

            # Compute Head1 on-target gradients
            v_on_pred_head1_log = v_on_pred_head1.log()

            on_grad_v_head1 = torch.autograd.grad(v_on_pred_head1, ligand_v_next,
                                                  grad_outputs=torch.ones_like(v_on_pred_head1),
                                                  retain_graph=True, allow_unused=True)[0]
            on_grad_pos_head1 = torch.autograd.grad(v_on_pred_head1_log, ligand_pos_original,
                                                    grad_outputs=torch.ones_like(v_on_pred_head1),
                                                    retain_graph=True, allow_unused=True)[0]

            if on_grad_v_head1 is None:
                on_grad_v_head1 = torch.zeros_like(ligand_v_next)
            if on_grad_pos_head1 is None:
                on_grad_pos_head1 = torch.zeros_like(ligand_pos_original)

            # ===== PHASE 2: HEAD2 (NO INTERACTION) ON-TARGET PREDICTION =====
            # Process ligand independently (NO protein interaction)
            h_ligand_no_int = self.ligand_atom_emb(ligand_v_next)
            if self.config.node_indicator:
                h_ligand_no_int = torch.cat([h_ligand_no_int, torch.ones(len(h_ligand_no_int), 1).to(h_ligand_no_int)], -1)

            mask_ligand_only = torch.ones(len(h_ligand_no_int), dtype=torch.bool, device=h_ligand_no_int.device)

            outputs_ligand = self.refine_net(
                h_ligand_no_int,
                ligand_pos_original,
                mask_ligand_only,
                batch_ligand,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            final_ligand_h_no_interaction = scatter_mean(outputs_ligand['h'], batch_ligand, dim=0)

            # Process on-target protein independently (NO ligand interaction)
            h_on_protein_no_int = self.protein_atom_emb(protein_v)
            if self.config.node_indicator:
                h_on_protein_no_int = torch.cat([h_on_protein_no_int, torch.zeros(len(h_on_protein_no_int), 1).to(h_on_protein_no_int)], -1)

            mask_protein_only = torch.zeros(len(h_on_protein_no_int), dtype=torch.bool, device=h_on_protein_no_int.device)

            outputs_on_protein = self.refine_net(
                h_on_protein_no_int,
                protein_pos,
                mask_protein_only,
                batch_protein,
                protein_id=None,
                return_all=False,
                fix_x=False
            )

            final_on_protein_h = scatter_mean(outputs_on_protein['h'], batch_protein, dim=0)

            # Predict on-target using Head2
            if final_ligand_h_no_interaction.size(0) != final_on_protein_h.size(0):
                final_on_protein_h = final_on_protein_h[0:1].expand(final_ligand_h_no_interaction.size(0), -1)

            # Check which variant of Head 2 we're using
            if hasattr(self, 'cross_attn_query'):
                # Atom-level cross-attention Head 2
                protein_h_atom = outputs_on_protein['h']
                ligand_h_atom = outputs_ligand['h']

                Q = self.cross_attn_query(protein_h_atom)
                K = self.cross_attn_key(ligand_h_atom)
                V = self.cross_attn_value(ligand_h_atom)

                Q = Q.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                K = K.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                V = V.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)

                batch_size_max = batch_ligand.max().item() + 1
                attended_protein_h_list = []

                for b in range(batch_size_max):
                    protein_mask_b = batch_protein == b if batch_protein.max().item() >= b else torch.ones_like(batch_protein, dtype=torch.bool)
                    ligand_mask_b = batch_ligand == b

                    Q_b = Q[protein_mask_b]
                    K_b = K[ligand_mask_b]
                    V_b = V[ligand_mask_b]

                    attn_scores_b = torch.einsum('phd,lhd->phl', Q_b, K_b) / np.sqrt(self.cross_attn_head_dim)
                    attn_weights_b = torch.softmax(attn_scores_b, dim=-1)

                    attended_b = torch.einsum('phl,lhd->phd', attn_weights_b, V_b)
                    attended_b = attended_b.reshape(-1, self.hidden_dim)

                    attended_b_mean = attended_b.mean(dim=0, keepdim=True)
                    attended_protein_h_list.append(attended_b_mean)

                h_mol_attended = torch.cat(attended_protein_h_list, dim=0)
                h_mol_attended = self.cross_attn_output(h_mol_attended)

                v_on_pred_head2 = head2_predictor(h_mol_attended)
            elif hasattr(self, 'non_interaction_query'):
                # Attention-based Head 2
                query = self.non_interaction_query(final_on_protein_h)
                key = self.non_interaction_key(final_ligand_h_no_interaction)
                value = self.non_interaction_value(final_ligand_h_no_interaction)

                attention_scores = torch.sum(query * key, dim=-1, keepdim=True) / self.non_interaction_scale

                if final_on_protein_h.size(0) == 1:
                    attention_weights = torch.sigmoid(attention_scores)
                else:
                    attention_weights = torch.softmax(attention_scores, dim=0)

                attended_features = attention_weights * value + (1 - attention_weights) * query

                v_on_pred_head2 = head2_predictor(attended_features)
            else:
                # Concatenation-based Head 2
                expert_input_on_head2 = torch.cat([final_ligand_h_no_interaction, final_on_protein_h], dim=1)
                v_on_pred_head2 = head2_predictor(expert_input_on_head2)

            # Compute Head2 on-target gradients
            v_on_pred_head2_log = v_on_pred_head2.log()

            on_grad_v_head2 = torch.autograd.grad(v_on_pred_head2, ligand_v_next,
                                                  grad_outputs=torch.ones_like(v_on_pred_head2),
                                                  retain_graph=True, allow_unused=True)[0]
            on_grad_pos_head2 = torch.autograd.grad(v_on_pred_head2_log, ligand_pos_original,
                                                    grad_outputs=torch.ones_like(v_on_pred_head2),
                                                    retain_graph=True, allow_unused=True)[0]

            if on_grad_v_head2 is None:
                on_grad_v_head2 = torch.zeros_like(ligand_v_next)
            if on_grad_pos_head2 is None:
                on_grad_pos_head2 = torch.zeros_like(ligand_pos_original)

            # ===== PHASE 3: HEAD2 (NO INTERACTION) OFF-TARGET PREDICTION =====
            v_off_pred_list = []

            if not off_target_data:
                v_off_pred_list = []
            else:
                for i, off_data in enumerate(off_target_data):
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)

                    if hasattr(off_data, 'protein_element_batch'):
                        off_batch_protein = off_data.protein_element_batch.to(protein_pos.device)
                    else:
                        off_batch_protein = torch.zeros(off_protein_pos.shape[0], dtype=torch.long, device=protein_pos.device)

                    if off_batch_protein.dim() > 1:
                        off_batch_protein = off_batch_protein.squeeze()

                    # Process off-target protein independently
                    h_off_protein = self.protein_atom_emb(off_protein_v)
                    if self.config.node_indicator:
                        h_off_protein = torch.cat([h_off_protein, torch.zeros(len(h_off_protein), 1).to(h_off_protein)], -1)

                    mask_off_protein_only = torch.zeros(len(h_off_protein), dtype=torch.bool, device=h_off_protein.device)

                    outputs_off_protein = self.refine_net(
                        h_off_protein,
                        off_protein_pos,
                        mask_off_protein_only,
                        off_batch_protein,
                        protein_id=None,
                        return_all=False,
                        fix_x=False
                    )

                    final_off_protein_h = scatter_mean(outputs_off_protein['h'], off_batch_protein, dim=0)

                    if final_ligand_h_no_interaction.size(0) != final_off_protein_h.size(0):
                        final_off_protein_h = final_off_protein_h[0:1].expand(final_ligand_h_no_interaction.size(0), -1)

                    # Predict using Head2
                    if hasattr(self, 'cross_attn_query'):
                        # Atom-level cross-attention
                        protein_h_atom_off = outputs_off_protein['h']
                        ligand_h_atom_off = outputs_ligand['h']

                        Q_off = self.cross_attn_query(protein_h_atom_off)
                        K_off = self.cross_attn_key(ligand_h_atom_off)
                        V_off = self.cross_attn_value(ligand_h_atom_off)

                        Q_off = Q_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                        K_off = K_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)
                        V_off = V_off.view(-1, self.num_cross_attn_heads, self.cross_attn_head_dim)

                        batch_size_max_off = batch_ligand.max().item() + 1
                        attended_protein_h_off_list = []

                        for b in range(batch_size_max_off):
                            protein_mask_b = off_batch_protein == b if off_batch_protein.max().item() >= b else torch.ones_like(off_batch_protein, dtype=torch.bool)
                            ligand_mask_b = batch_ligand == b

                            Q_off_b = Q_off[protein_mask_b]
                            K_off_b = K_off[ligand_mask_b]
                            V_off_b = V_off[ligand_mask_b]

                            attn_scores_off_b = torch.einsum('phd,lhd->phl', Q_off_b, K_off_b) / np.sqrt(self.cross_attn_head_dim)
                            attn_weights_off_b = torch.softmax(attn_scores_off_b, dim=-1)

                            attended_off_b = torch.einsum('phl,lhd->phd', attn_weights_off_b, V_off_b)
                            attended_off_b = attended_off_b.reshape(-1, self.hidden_dim)

                            attended_off_b_mean = attended_off_b.mean(dim=0, keepdim=True)
                            attended_protein_h_off_list.append(attended_off_b_mean)

                        h_mol_attended_off = torch.cat(attended_protein_h_off_list, dim=0)
                        h_mol_attended_off = self.cross_attn_output(h_mol_attended_off)

                        v_off_pred = head2_predictor(h_mol_attended_off)
                    elif hasattr(self, 'non_interaction_query'):
                        # Attention-based Head 2
                        query_off = self.non_interaction_query(final_off_protein_h)
                        key_off = self.non_interaction_key(final_ligand_h_no_interaction)
                        value_off = self.non_interaction_value(final_ligand_h_no_interaction)

                        attention_scores_off = torch.sum(query_off * key_off, dim=-1, keepdim=True) / self.non_interaction_scale

                        if final_off_protein_h.size(0) == 1:
                            attention_weights_off = torch.sigmoid(attention_scores_off)
                        else:
                            attention_weights_off = torch.softmax(attention_scores_off, dim=0)

                        attended_features_off = attention_weights_off * value_off + (1 - attention_weights_off) * query_off

                        v_off_pred = head2_predictor(attended_features_off)
                    else:
                        # Concatenation-based Head 2
                        expert_input_off = torch.cat([final_ligand_h_no_interaction, final_off_protein_h], dim=1)
                        v_off_pred = head2_predictor(expert_input_off)

                    v_off_pred_list.append(v_off_pred)

            # Combine off-target predictions
            if len(v_off_pred_list) > 0:
                v_off_pred = torch.cat(v_off_pred_list, dim=1)
                v_off_mean_head2 = torch.mean(v_off_pred, dim=1, keepdim=True)
            else:
                v_off_pred = torch.empty(v_on_pred_head2.shape[0], 0, device=v_on_pred_head2.device)
                v_off_mean_head2 = torch.zeros_like(v_on_pred_head2)

            # Compute Head2 off-target gradients
            if v_off_mean_head2.sum() > 0:
                v_off_mean_head2_log = v_off_mean_head2.log()

                off_grad_v_head2 = torch.autograd.grad(v_off_mean_head2, ligand_v_next,
                                                       grad_outputs=torch.ones_like(v_off_mean_head2),
                                                       retain_graph=True, allow_unused=True)[0]
                off_grad_pos_head2 = torch.autograd.grad(v_off_mean_head2_log, ligand_pos_original,
                                                         grad_outputs=torch.ones_like(v_off_mean_head2),
                                                         retain_graph=False, allow_unused=True)[0]

                if off_grad_v_head2 is None:
                    off_grad_v_head2 = torch.zeros_like(ligand_v_next)
                if off_grad_pos_head2 is None:
                    off_grad_pos_head2 = torch.zeros_like(ligand_pos_original)
            else:
                off_grad_v_head2 = torch.zeros_like(ligand_v_next)
                off_grad_pos_head2 = torch.zeros_like(ligand_pos_original)

        # Return all gradients and predictions
        return (on_grad_v_head1, on_grad_pos_head1,
                on_grad_v_head2, on_grad_pos_head2,
                off_grad_v_head2, off_grad_pos_head2,
                v_on_pred_head1, v_on_pred_head2, v_off_mean_head2)

    # @torch.no_grad()
    def sample_diffusion(self, guide_mode, type_grad_weight, pos_grad_weight, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand, protein_id=None, value_model=None,
                         num_steps=None, center_pos_mode=None, w_off=1.0, on_target_only=False,
                         off_target_data=None, w_on=1.0,
                         head1_type_grad_weight=None, head1_pos_grad_weight=None,
                         head2_type_grad_weight=None, head2_pos_grad_weight=None):
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        # Default head-specific weights to general weights if not specified
        if head1_type_grad_weight is None:
            head1_type_grad_weight = type_grad_weight
        if head1_pos_grad_weight is None:
            head1_pos_grad_weight = pos_grad_weight
        if head2_type_grad_weight is None:
            head2_type_grad_weight = type_grad_weight
        if head2_pos_grad_weight is None:
            head2_pos_grad_weight = pos_grad_weight

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode, center_ligand=self.center_ligand)

        pos_traj, v_traj, exp_traj, exp_atom_traj = [], [], [], []
        exp_off_traj = []  # Track off-target predictions
        exp_on_traj = []   # Track on-target predictions
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))

        # DEBUG: Print at the very start to confirm function is called
        print(f"\n{'='*80}")
        print(f"[SAMPLE_DIFFUSION] Starting sampling with guide_mode='{guide_mode}'")
        print(f"[SAMPLE_DIFFUSION] num_steps={num_steps}, w_on={w_on}, w_off={w_off}")
        print(f"[SAMPLE_DIFFUSION] General weights: type_grad={type_grad_weight}, pos_grad={pos_grad_weight}")
        print(f"[SAMPLE_DIFFUSION] Head1 weights: type_grad={head1_type_grad_weight}, pos_grad={head1_pos_grad_weight}")
        print(f"[SAMPLE_DIFFUSION] Head2 weights: type_grad={head2_type_grad_weight}, pos_grad={head2_pos_grad_weight}")
        print(f"[SAMPLE_DIFFUSION] off_target_data type: {type(off_target_data)}")
        if off_target_data is not None:
            if isinstance(off_target_data, list):
                print(f"[SAMPLE_DIFFUSION] Number of off-targets: {len(off_target_data)}")
            else:
                print(f"[SAMPLE_DIFFUSION] off_target_data is not a list")
        print(f"{'='*80}\n")

        for iter_idx, i in enumerate(tqdm(time_seq, desc='sampling', total=len(time_seq))):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random':
                # Handle protein_id for selectivity guidance
                if protein_id is None:
                    # Create dummy protein_id for backward compatibility 
                    protein_id = torch.zeros_like(batch_protein)
                preds, type_grad, pos_grad = self.pv_joint_guide(
                    ligand_v, ligand_pos, protein_v, protein_pos, 
                    batch_protein, batch_ligand, protein_id, w_off=w_off, on_target_only=on_target_only
                )
            elif guide_mode == 'selectivity':
                # Selectivity mode: Use combined multi-protein batch (like training)
                if protein_id is None:
                    protein_id = torch.zeros_like(batch_protein)
                
                # Use the combined batch with multi-protein data
                # The batch already contains on-target (protein_id=0) and off-targets (protein_id=1,2,3...)
                preds, type_grad, pos_grad = self.pv_joint_guide(
                    ligand_v, ligand_pos, protein_v, protein_pos, 
                    batch_protein, batch_ligand, protein_id, w_off=w_off, on_target_only=on_target_only
                )
                
                # Track off-target predictions (v_off_pred from the multi-protein forward pass)
                if 'v_off_pred' in preds and preds['v_off_pred'].numel() > 0:
                    # Handle both 1D [batch_size] (single off-target) and 2D [batch_size, n] (multiple off-targets)
                    if preds['v_off_pred'].dim() == 1:
                        off_exp_pred = preds['v_off_pred'].unsqueeze(1)  # [batch_size, 1]
                    else:
                        off_exp_pred = preds['v_off_pred'].mean(dim=1, keepdim=True)  # Average across off-targets
                    exp_off_traj.append(off_exp_pred.clone().cpu())

            elif guide_mode == 'sequential_selectivity':
                # Sequential selectivity mode: Use separate calls for on-target and off-targets
                if off_target_data is None:
                    raise ValueError("off_target_data is required for sequential_selectivity mode")
                
                preds, type_grad, pos_grad = self.sequential_selectivity_guide(
                    ligand_v, ligand_pos, protein_v, protein_pos, 
                    batch_protein, batch_ligand, off_target_data, t, 
                    w_on=w_on, w_off=w_off
                )
                # Track predictions
                if 'v_off_pred' in preds and preds['v_off_pred'].numel() > 0:
                    exp_off_traj.append(preds['v_off_pred'].clone().cpu())

            elif guide_mode == 'target_diff':
                preds = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )
                pred = value_model(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,

                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )
            
            else:
                # For head2_only_sequential, we don't need multi-protein forward
                # Use simple forward with protein_id=0 for on-target only
                if guide_mode == 'head2_only_sequential':
                    # Create protein_id for on-target (all zeros)
                    if protein_id is None:
                        protein_id = torch.zeros_like(batch_protein)
                    preds = self(
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        protein_id=protein_id,
                        time_step=t
                    )
                else:
                    preds = self(
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t,
                        protein_id=protein_id
                    )
                
            # Compute posterior mean and variance
            # Automatic key mapping for pred_ligand_pos (x0 reconstruction)
            if 'pred_ligand_pos' not in preds:
                # Try various keys that may contain x0 reconstruction
                if 'ligand_pos' in preds:
                    # Forward pass returns 'ligand_pos' as x0 reconstruction
                    preds['pred_ligand_pos'] = preds['ligand_pos']
                elif 'final_ligand_pos' in preds:
                    preds['pred_ligand_pos'] = preds['final_ligand_pos']
                elif 'pos' in preds:
                    preds['pred_ligand_pos'] = preds['pos']
                else:
                    raise KeyError(f"'pred_ligand_pos' not found in preds and no suitable fallback available. Keys: {list(preds.keys())}")

            if 'pred_ligand_v' not in preds:
                # Try various keys that may contain v0 reconstruction
                if 'ligand_v' in preds:
                    # Forward pass returns 'ligand_v' as v0 reconstruction
                    preds['pred_ligand_v'] = preds['ligand_v']
                elif 'final_ligand_v' in preds:
                    preds['pred_ligand_v'] = preds['final_ligand_v']
                else:
                    raise KeyError(f"'pred_ligand_v' not found in preds and no suitable fallback available. Keys: {list(preds.keys())}")

            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            # pos posterior
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)

            # type posterior
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
            
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random':
                exp_pred = preds.get('selectivity_score', preds.get('final_exp_pred', None))
                
                # Apply selectivity-based gradients to guide molecular generation
                # Position guidance: Push coordinates toward higher selectivity
                pos_model_mean = pos_model_mean + pos_grad_weight*(0.5 * pos_log_variance).exp()*pos_grad
                # Atom type guidance: Push atom types toward higher selectivity  
                log_ligand_v = log_ligand_v + type_grad_weight*type_grad
                
                # Debug: Print gradient magnitudes
                if i % 200 == 0:  # Print every 200 steps to avoid spam
                    # print(f"[DEBUG Step {i}] Pos grad magnitude: {torch.norm(pos_grad).item():.6f}")
                    # print(f"[DEBUG Step {i}] Type grad magnitude: {torch.norm(type_grad).item():.6f}")
                    # print(f"[DEBUG Step {i}] Pos grad weight: {pos_grad_weight}, Type grad weight: {type_grad_weight}")
                    pass
                
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                
            elif guide_mode == 'vina':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.vina_classifier_gradient(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t)
                ligand_v_next_prob = ligand_v_next_prob - grads1 * type_grad_weight
                ligand_pos = (ligand_pos - grads2 * pos_grad_weight).detach()  # Break computation graph to save memory
                
            
            elif guide_mode == 'target_diff':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                exp_pred = pred['final_exp_pred']
                preds['atom_affinity'] = pred['atom_affinity']
                
            elif guide_mode == 'valuenet_rep' or guide_mode == 'valuenet_rep1':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.value_net_classifier_gradient_rep(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t,value_model)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = (ligand_pos + grads2 * pos_grad_weight).detach()  # Break computation graph to save memory
            
            elif guide_mode == 'valuenet_rep2':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.value_net_classifier_gradient_rep2(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t,value_model)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = (ligand_pos + grads2 * pos_grad_weight).detach()  # Break computation graph to save memory
                   
            elif guide_mode == 'selectivity':
                exp_pred = preds.get('selectivity_score', preds.get('final_exp_pred', None))
                
                # Algorithm: x_{t-1}: μ̃_t + s_pos * β̃_t * grad_x 기반 샘플링
                # β̃_t = (0.5 * pos_log_variance).exp() = posterior std
                pos_model_mean = pos_model_mean + pos_grad_weight * (0.5 * pos_log_variance).exp() * pos_grad
                
                # Algorithm: z_{t-1}: log(C) + s_type * grad_z 기반 categorical 샘플링  
                # log_ligand_v는 현재 z_t의 log probability
                log_ligand_v = log_ligand_v + type_grad_weight * type_grad
                
                # Position sampling: x_{t-1} ~ N(μ̃_t + guidance, β̃_t)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                
                # Type sampling: use guided log probabilities
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
            elif guide_mode == 'sequential_selectivity':
                exp_pred = preds.get('selectivity_score', preds.get('final_exp_pred', None))
                
                # Apply selectivity-based gradients to guide molecular generation
                # Position guidance: Push coordinates toward higher selectivity
                pos_model_mean = pos_model_mean + pos_grad_weight * (0.5 * pos_log_variance).exp() * pos_grad
                # Atom type guidance: Push atom types toward higher selectivity  
                log_ligand_v = log_ligand_v + type_grad_weight * type_grad
                
                # Position sampling: x_{t-1} ~ N(μ̃_t + guidance, β̃_t)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                
                # Type sampling: z_{t-1} ~ Categorical(softmax(log(C) + guidance))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                # Track off-target predictions
                if 'v_off_pred' in preds and preds['v_off_pred'].numel() > 0:
                    exp_off_traj.append(preds['v_off_pred'].clone().cpu())
                    
            elif guide_mode == 'valuenet_sequential_selectivity':
                # Valuenet-based sequential selectivity guidance
                if value_model is None:
                    raise ValueError("value_model is required for valuenet_sequential_selectivity mode")
                if off_target_data is None:
                    raise ValueError("off_target_data is required for valuenet_sequential_selectivity mode")
                
                # Position sampling first (like original valuenet)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                
                # Type sampling
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                # Get selectivity gradients using valuenet
                grads1, grads2, exp_pred, v_on_pred, v_off_pred = self.valuenet_sequential_selectivity_guide(
                    ligand_v_next_prob, ligand_pos, protein_v, protein_pos, 
                    batch_protein, batch_ligand, off_target_data, t, value_model,
                    w_on=w_on, w_off=w_off
                )
                # Apply gradients (like original valuenet)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = ligand_pos + grads2 * pos_grad_weight
                
                # Track predictions
                exp_off_traj.append(v_off_pred.clone().cpu())
            
            elif guide_mode == 'valuenet':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                
                grads1, grads2, exp_pred = self.value_net_classifier_gradient(ligand_v_next_prob, ligand_pos, protein_v, protein_pos, batch_ligand, batch_protein, t,value_model)
                ligand_v_next_prob = ligand_v_next_prob + grads1 * type_grad_weight
                ligand_pos = (ligand_pos + grads2 * pos_grad_weight).detach()  # Break computation graph to save memory


            elif guide_mode == 'pretrained_joint_aligned':
                # Joint model-based sequential selectivity guidance (memory efficient)
                if off_target_data is None:
                    raise ValueError("off_target_data is required for pretrained_joint_aligned mode")

                # Get selectivity gradients using joint model (similar to joint mode)
                grads1, grads2, exp_pred, v_on_pred, v_off_pred = self.joint_sequential_selectivity_guide(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand, off_target_data, t,
                    w_on=w_on, w_off=w_off
                )

                # Apply gradients to posterior mean (like joint mode)
                # Position guidance: Push coordinates toward higher selectivity
                pos_model_mean = pos_model_mean + pos_grad_weight*(0.5 * pos_log_variance).exp()*grads2
                # Atom type guidance: Push atom types toward higher selectivity
                log_ligand_v = log_ligand_v + type_grad_weight*grads1

                # Single sampling after gradient application (like joint mode)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)

                # Track predictions
                exp_off_traj.append(v_off_pred.clone().cpu())

            elif guide_mode == 'pretrained_joint_no_off_interaction':
                # NEW: Joint model-based sequential selectivity guidance WITHOUT off-target interaction
                # Off-target protein processed independently (no ligand involved)
                if off_target_data is None:
                    raise ValueError("off_target_data is required for pretrained_joint_no_off_interaction mode")

                # Get selectivity gradients using NEW no-interaction guidance
                grads1, grads2, exp_pred, v_on_pred, v_off_pred = self.joint_sequential_selectivity_guide_no_interaction(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand, off_target_data, t,
                    w_on=w_on, w_off=w_off
                )

                # Apply gradients to posterior mean (like joint mode)
                # Position guidance: Push coordinates toward higher selectivity
                pos_model_mean = pos_model_mean + pos_grad_weight*(0.5 * pos_log_variance).exp()*grads2
                # Atom type guidance: Push atom types toward higher selectivity
                log_ligand_v = log_ligand_v + type_grad_weight*grads1

                # Single sampling after gradient application (like joint mode)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)

                # Track predictions
                exp_off_traj.append(v_off_pred.clone().cpu())

            elif guide_mode == 'joint_on_off_no_interaction_sequential':
                # DEBUG: Confirm this path is executed
                if iter_idx == 0:
                    print(f"\n*** ENTERING joint_on_off_no_interaction_sequential MODE ***")
                    print(f"*** Using Head1 (interaction) for on-target, Head2 (no-interaction) for off-target ***\n")

                # NEW SEQUENTIAL STRATEGY: Apply on-target and off-target gradients sequentially
                # This solves the gradient cancellation problem
                # Allow None or empty list for on-target only mode
                if off_target_data is None:
                    off_target_data = []  # Convert None to empty list

                # Get SEPARATE on-target and off-target gradients
                on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_pred = self.joint_on_off_no_interaction_sequential(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand, off_target_data, t,
                    w_on=w_on, w_off=w_off
                )

                # SEQUENTIAL APPLICATION (key difference from combined selectivity)
                # Step 1: Apply on-target guidance (Head1) to posterior mean (PRIMARY)
                # Use head1_pos_grad_weight for on-target (interaction-based)
                pos_model_mean_on = pos_model_mean + w_on * head1_pos_grad_weight * (0.5 * pos_log_variance).exp() * on_grad_pos

                # Debug: Print gradient magnitudes every 200 iterations
                # Note: iter_idx is 0, 1, 2, ..., 999; i is timestep 999, 998, ..., 0
                if iter_idx % 200 == 0 or iter_idx == 0:
                    debug_msg = f"\n[DEBUG Joint Iter {iter_idx}, Timestep {i}] Gradient Magnitudes:\n"
                    debug_msg += f"  Head1 weights (on-target): type_grad={head1_type_grad_weight}, pos_grad={head1_pos_grad_weight}\n"
                    debug_msg += f"  Head2 weights (off-target): type_grad={head2_type_grad_weight}, pos_grad={head2_pos_grad_weight}\n"
                    debug_msg += f"  On-target  pos grad: {torch.norm(on_grad_pos).item():.6f}\n"
                    debug_msg += f"  On-target  type grad: {torch.norm(on_grad_v).item():.6f}\n"
                    debug_msg += f"  Off-target pos grad: {torch.norm(off_grad_pos).item():.6f}\n"
                    debug_msg += f"  Off-target type grad: {torch.norm(off_grad_v).item():.6f}\n"

                    if v_off_pred.numel() > 0:
                        debug_msg += f"  v_on_pred (head1): {v_on_pred.mean().item():.4f}, v_off_pred (head2): {v_off_pred.mean().item():.4f}\n"
                        debug_msg += f"  Selectivity: {(v_on_pred - v_off_pred).mean().item():.4f}\n"
                    else:
                        debug_msg += f"  v_on_pred (head1): {v_on_pred.mean().item():.4f}, v_off_pred: N/A (on-target only)\n"

                    print(debug_msg, end='')

                    # Save to debug log file if available
                    if hasattr(self, '_debug_log_file') and self._debug_log_file is not None:
                        try:
                            with open(self._debug_log_file, 'a') as f:
                                f.write(debug_msg)
                        except Exception as e:
                            print(f"[WARNING] Failed to write to debug log: {e}")

                # Debug: Check gradient shapes
                if on_grad_v.shape != log_ligand_v.shape:
                    print(f"WARNING: Shape mismatch - on_grad_v: {on_grad_v.shape}, log_ligand_v: {log_ligand_v.shape}")
                    print(f"  ligand_v shape: {ligand_v.shape}")
                    print(f"  on_grad_v should be [num_atoms, num_classes], got {on_grad_v.shape}")
                    # Attempt to fix: if on_grad_v is wrong shape, reshape it
                    if on_grad_v.dim() == 1 and log_ligand_v.dim() == 2:
                        # on_grad_v is [num_classes], expand to [num_atoms, num_classes]
                        on_grad_v = on_grad_v.unsqueeze(0).expand_as(log_ligand_v)
                        print(f"  Reshaped on_grad_v to {on_grad_v.shape}")

                # Use head1_type_grad_weight for on-target (interaction-based)
                log_ligand_v_on = log_ligand_v + w_on * head1_type_grad_weight * on_grad_v

                # Step 2: Apply off-target guidance (Head2) to REFINED state (SECONDARY)
                # Note: off-target gradient pushes AWAY from off-target binding (negative sign)
                # Use head2_pos_grad_weight for off-target (no-interaction)
                pos_model_mean_final = pos_model_mean_on - w_off * head2_pos_grad_weight * (0.5 * pos_log_variance).exp() * off_grad_pos

                # Debug: Check off-target gradient shapes
                if off_grad_v.shape != log_ligand_v_on.shape:
                    print(f"WARNING: Shape mismatch - off_grad_v: {off_grad_v.shape}, log_ligand_v_on: {log_ligand_v_on.shape}")
                    # Attempt to fix
                    if off_grad_v.dim() == 1 and log_ligand_v_on.dim() == 2:
                        off_grad_v = off_grad_v.unsqueeze(0).expand_as(log_ligand_v_on)
                        print(f"  Reshaped off_grad_v to {off_grad_v.shape}")

                # Use head2_type_grad_weight for off-target (no-interaction)
                log_ligand_v_final = log_ligand_v_on - w_off * head2_type_grad_weight * off_grad_v

                # Single sampling after sequential gradient application
                ligand_pos_next = pos_model_mean_final + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_final, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)

                # Track predictions (use selectivity as exp_pred for monitoring)
                exp_pred = v_on_pred - v_off_pred  # Selectivity score
                exp_on_traj.append(v_on_pred.clone().cpu())
                exp_off_traj.append(v_off_pred.clone().cpu())

            elif guide_mode == 'head1_only_sequential':
                # DEBUG: Confirm this path is executed
                if iter_idx == 0:
                    print(f"\n*** ENTERING head1_only_sequential MODE ***")
                    print(f"*** Using Head1 (Interaction-Based) for both on-target and off-target predictions ***\n")

                # NEW HEAD1-ONLY SEQUENTIAL STRATEGY
                # Allow None or empty list for on-target only mode
                if off_target_data is None:
                    off_target_data = []  # Convert None to empty list

                # Get SEPARATE on-target and off-target gradients using head1 only
                on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_pred = self.head1_only_sequential_guidance(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand, off_target_data, t,
                    w_on=w_on, w_off=w_off
                )

                # SEQUENTIAL APPLICATION (key difference from combined selectivity)
                # Step 1: Apply on-target guidance to posterior mean (PRIMARY) - Use head1_pos_grad_weight
                pos_model_mean_on = pos_model_mean + w_on * head1_pos_grad_weight * (0.5 * pos_log_variance).exp() * on_grad_pos

                # Debug: Print gradient magnitudes every 200 iterations
                if iter_idx % 200 == 0 or iter_idx == 0:
                    debug_msg = f"\n[DEBUG Head1 Iter {iter_idx}, Timestep {i}] Gradient Magnitudes:\n"
                    debug_msg += f"  Head1 weights: type_grad={head1_type_grad_weight}, pos_grad={head1_pos_grad_weight}\n"
                    debug_msg += f"  On-target  pos grad: {torch.norm(on_grad_pos).item():.6f}\n"
                    debug_msg += f"  On-target  type grad: {torch.norm(on_grad_v).item():.6f}\n"
                    debug_msg += f"  Off-target pos grad: {torch.norm(off_grad_pos).item():.6f}\n"
                    debug_msg += f"  Off-target type grad: {torch.norm(off_grad_v).item():.6f}\n"

                    if v_off_pred.numel() > 0:
                        debug_msg += f"  v_on_pred (head1): {v_on_pred.mean().item():.4f}, v_off_pred (head1): {v_off_pred.mean().item():.4f}\n"
                        debug_msg += f"  Selectivity: {(v_on_pred - v_off_pred).mean().item():.4f}\n"
                    else:
                        debug_msg += f"  v_on_pred (head1): {v_on_pred.mean().item():.4f}, v_off_pred: N/A (on-target only)\n"

                    print(debug_msg, end='')

                    # Save to debug log file if available
                    if hasattr(self, '_debug_log_file') and self._debug_log_file is not None:
                        try:
                            with open(self._debug_log_file, 'a') as f:
                                f.write(debug_msg)
                        except Exception as e:
                            print(f"[WARNING] Failed to write to debug log: {e}")

                # Debug: Check gradient shapes
                if on_grad_v.shape != log_ligand_v.shape:
                    print(f"WARNING: Shape mismatch - on_grad_v: {on_grad_v.shape}, log_ligand_v: {log_ligand_v.shape}")
                    if on_grad_v.dim() == 1 and log_ligand_v.dim() == 2:
                        on_grad_v = on_grad_v.unsqueeze(0).expand_as(log_ligand_v)
                        print(f"  Reshaped on_grad_v to {on_grad_v.shape}")

                # Use head1_type_grad_weight for head1 mode
                log_ligand_v_on = log_ligand_v + w_on * head1_type_grad_weight * on_grad_v

                # Step 2: Apply off-target guidance to REFINED state (SECONDARY) - Use head1_pos_grad_weight
                pos_model_mean_final = pos_model_mean_on - w_off * head1_pos_grad_weight * (0.5 * pos_log_variance).exp() * off_grad_pos

                # Debug: Check off_grad_v shape
                if off_grad_v.shape != log_ligand_v_on.shape:
                    print(f"WARNING: Shape mismatch - off_grad_v: {off_grad_v.shape}, log_ligand_v_on: {log_ligand_v_on.shape}")
                    if off_grad_v.dim() == 1 and log_ligand_v_on.dim() == 2:
                        off_grad_v = off_grad_v.unsqueeze(0).expand_as(log_ligand_v_on)
                        print(f"  Reshaped off_grad_v to {off_grad_v.shape}")

                # Use head1_type_grad_weight for head1 mode
                log_ligand_v_final = log_ligand_v_on - w_off * head1_type_grad_weight * off_grad_v

                # Single sampling after sequential gradient application
                ligand_pos_next = pos_model_mean_final + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_final, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)

                # Track predictions (use selectivity as exp_pred for monitoring)
                exp_pred = v_on_pred - v_off_pred  # Selectivity score
                exp_on_traj.append(v_on_pred.clone().cpu())
                exp_off_traj.append(v_off_pred.clone().cpu())

            elif guide_mode == 'head2_only_sequential':
                # DEBUG: Confirm this path is executed
                if iter_idx == 0:
                    print(f"\n*** ENTERING head2_only_sequential MODE ***")
                    print(f"*** Using Head2 (No-Interaction) for both on-target and off-target predictions ***\n")

                # NEW HEAD2-ONLY SEQUENTIAL STRATEGY
                # Allow None or empty list for on-target only mode
                if off_target_data is None:
                    off_target_data = []  # Convert None to empty list

                # Get SEPARATE on-target and off-target gradients using head2 only
                on_grad_v, on_grad_pos, off_grad_v, off_grad_pos, v_on_pred, v_off_pred = self.head2_only_sequential_guidance(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand, off_target_data, t,
                    w_on=w_on, w_off=w_off
                )

                # SEQUENTIAL APPLICATION (key difference from combined selectivity)
                # Step 1: Apply on-target guidance to posterior mean (PRIMARY) - Use head2_pos_grad_weight
                pos_model_mean_on = pos_model_mean + w_on * head2_pos_grad_weight * (0.5 * pos_log_variance).exp() * on_grad_pos

                # Debug: Print gradient magnitudes every 200 iterations
                if iter_idx % 200 == 0 or iter_idx == 0:
                    debug_msg = f"\n[DEBUG Head2 Iter {iter_idx}, Timestep {i}] Gradient Magnitudes:\n"
                    debug_msg += f"  Head2 weights: type_grad={head2_type_grad_weight}, pos_grad={head2_pos_grad_weight}\n"
                    debug_msg += f"  On-target  pos grad: {torch.norm(on_grad_pos).item():.6f}\n"
                    debug_msg += f"  On-target  type grad: {torch.norm(on_grad_v).item():.6f}\n"
                    debug_msg += f"  Off-target pos grad: {torch.norm(off_grad_pos).item():.6f}\n"
                    debug_msg += f"  Off-target type grad: {torch.norm(off_grad_v).item():.6f}\n"

                    if v_off_pred.numel() > 0:
                        debug_msg += f"  v_on_pred (head2): {v_on_pred.mean().item():.4f}, v_off_pred (head2): {v_off_pred.mean().item():.4f}\n"
                        debug_msg += f"  Selectivity: {(v_on_pred - v_off_pred).mean().item():.4f}\n"
                    else:
                        debug_msg += f"  v_on_pred (head2): {v_on_pred.mean().item():.4f}, v_off_pred: N/A (on-target only)\n"

                    print(debug_msg, end='')

                    # Save to debug log file if available
                    if hasattr(self, '_debug_log_file') and self._debug_log_file is not None:
                        try:
                            with open(self._debug_log_file, 'a') as f:
                                f.write(debug_msg)
                        except Exception as e:
                            print(f"[WARNING] Failed to write to debug log: {e}")

                # Debug: Check gradient shapes
                if on_grad_v.shape != log_ligand_v.shape:
                    print(f"WARNING: Shape mismatch - on_grad_v: {on_grad_v.shape}, log_ligand_v: {log_ligand_v.shape}")
                    if on_grad_v.dim() == 1 and log_ligand_v.dim() == 2:
                        on_grad_v = on_grad_v.unsqueeze(0).expand_as(log_ligand_v)
                        print(f"  Reshaped on_grad_v to {on_grad_v.shape}")

                # Use head2_type_grad_weight for head2 mode
                log_ligand_v_on = log_ligand_v + w_on * head2_type_grad_weight * on_grad_v

                # Step 2: Apply off-target guidance to REFINED state (SECONDARY) - Use head2_pos_grad_weight
                pos_model_mean_final = pos_model_mean_on - w_off * head2_pos_grad_weight * (0.5 * pos_log_variance).exp() * off_grad_pos

                # Debug: Check off_grad_v shape
                if off_grad_v.shape != log_ligand_v_on.shape:
                    print(f"WARNING: Shape mismatch - off_grad_v: {off_grad_v.shape}, log_ligand_v_on: {log_ligand_v_on.shape}")
                    if off_grad_v.dim() == 1 and log_ligand_v_on.dim() == 2:
                        off_grad_v = off_grad_v.unsqueeze(0).expand_as(log_ligand_v_on)
                        print(f"  Reshaped off_grad_v to {off_grad_v.shape}")

                # Use head2_type_grad_weight for head2 mode
                log_ligand_v_final = log_ligand_v_on - w_off * head2_type_grad_weight * off_grad_v

                # Single sampling after sequential gradient application
                ligand_pos_next = pos_model_mean_final + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_final, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)

                # Track predictions (use selectivity as exp_pred for monitoring)
                exp_pred = v_on_pred - v_off_pred  # Selectivity score
                exp_on_traj.append(v_on_pred.clone().cpu())
                exp_off_traj.append(v_off_pred.clone().cpu())

            elif guide_mode == 'head1_head2_sequential':
                # DEBUG: Confirm this path is executed
                if iter_idx == 0:
                    print(f"\n*** ENTERING head1_head2_sequential MODE ***")
                    print(f"*** Using Head1 (Interaction) for on-target, Head2 (No-Interaction) for on/off-target ***\n")

                # NEW HEAD1-HEAD2 SEQUENTIAL STRATEGY
                if off_target_data is None:
                    off_target_data = []  # Convert None to empty list

                # Get ALL gradients: head1 on-target, head2 on-target, head2 off-target
                (on_grad_v_head1, on_grad_pos_head1,
                 on_grad_v_head2, on_grad_pos_head2,
                 off_grad_v_head2, off_grad_pos_head2,
                 v_on_pred_head1, v_on_pred_head2, v_off_pred_head2) = self.head1_head2_sequential_guidance(
                    ligand_v, ligand_pos, protein_v, protein_pos,
                    batch_protein, batch_ligand, off_target_data, t,
                    w_on=w_on, w_off=w_off
                )

                # PHASE 1: Apply Head1 on-target guidance (WITH interaction)
                # Step 1: Apply head1 on-target position guidance
                pos_model_mean_h1 = pos_model_mean + w_on * head1_pos_grad_weight * (0.5 * pos_log_variance).exp() * on_grad_pos_head1

                # Debug: Print gradient magnitudes every 200 iterations
                if iter_idx % 200 == 0 or iter_idx == 0:
                    debug_msg = f"\n[DEBUG Head1+Head2 Iter {iter_idx}, Timestep {i}] Gradient Magnitudes:\n"
                    debug_msg += f"  Head1 weights: type_grad={head1_type_grad_weight}, pos_grad={head1_pos_grad_weight}\n"
                    debug_msg += f"  Head2 weights: type_grad={head2_type_grad_weight}, pos_grad={head2_pos_grad_weight}\n"
                    debug_msg += f"  Head1 On-target  pos grad: {torch.norm(on_grad_pos_head1).item():.6f}\n"
                    debug_msg += f"  Head1 On-target  type grad: {torch.norm(on_grad_v_head1).item():.6f}\n"
                    debug_msg += f"  Head2 On-target  pos grad: {torch.norm(on_grad_pos_head2).item():.6f}\n"
                    debug_msg += f"  Head2 On-target  type grad: {torch.norm(on_grad_v_head2).item():.6f}\n"
                    debug_msg += f"  Head2 Off-target pos grad: {torch.norm(off_grad_pos_head2).item():.6f}\n"
                    debug_msg += f"  Head2 Off-target type grad: {torch.norm(off_grad_v_head2).item():.6f}\n"

                    debug_msg += f"  v_on_pred (head1): {v_on_pred_head1.mean().item():.4f}\n"
                    debug_msg += f"  v_on_pred (head2): {v_on_pred_head2.mean().item():.4f}\n"
                    if v_off_pred_head2.numel() > 0:
                        debug_msg += f"  v_off_pred (head2): {v_off_pred_head2.mean().item():.4f}\n"
                        debug_msg += f"  Selectivity (head2): {(v_on_pred_head2 - v_off_pred_head2).mean().item():.4f}\n"
                    else:
                        debug_msg += f"  v_off_pred (head2): N/A (on-target only)\n"

                    print(debug_msg, end='')

                    # Save to debug log file if available
                    if hasattr(self, '_debug_log_file') and self._debug_log_file is not None:
                        try:
                            with open(self._debug_log_file, 'a') as f:
                                f.write(debug_msg)
                        except Exception as e:
                            print(f"[WARNING] Failed to write to debug log: {e}")

                # Check gradient shapes for head1
                if on_grad_v_head1.shape != log_ligand_v.shape:
                    if on_grad_v_head1.dim() == 1 and log_ligand_v.dim() == 2:
                        on_grad_v_head1 = on_grad_v_head1.unsqueeze(0).expand_as(log_ligand_v)

                # Step 2: Apply head1 on-target type guidance
                log_ligand_v_h1 = log_ligand_v + w_on * head1_type_grad_weight * on_grad_v_head1

                # PHASE 2: Apply Head2 on-target guidance (NO interaction)
                # Step 3: Apply head2 on-target position guidance to head1-refined state
                pos_model_mean_h2 = pos_model_mean_h1 + w_on * head2_pos_grad_weight * (0.5 * pos_log_variance).exp() * on_grad_pos_head2

                # Check gradient shapes for head2 on-target
                if on_grad_v_head2.shape != log_ligand_v_h1.shape:
                    if on_grad_v_head2.dim() == 1 and log_ligand_v_h1.dim() == 2:
                        on_grad_v_head2 = on_grad_v_head2.unsqueeze(0).expand_as(log_ligand_v_h1)

                # Step 4: Apply head2 on-target type guidance to head1-refined type
                log_ligand_v_h2 = log_ligand_v_h1 + w_on * head2_type_grad_weight * on_grad_v_head2

                # PHASE 3: Apply Head2 off-target guidance (NO interaction)
                # Step 5: Apply head2 off-target position guidance (repulsive) to head2-refined state
                pos_model_mean_final = pos_model_mean_h2 - w_off * head2_pos_grad_weight * (0.5 * pos_log_variance).exp() * off_grad_pos_head2

                # Check gradient shapes for head2 off-target
                if off_grad_v_head2.shape != log_ligand_v_h2.shape:
                    if off_grad_v_head2.dim() == 1 and log_ligand_v_h2.dim() == 2:
                        off_grad_v_head2 = off_grad_v_head2.unsqueeze(0).expand_as(log_ligand_v_h2)

                # Step 6: Apply head2 off-target type guidance (repulsive) to head2-refined type
                log_ligand_v_final = log_ligand_v_h2 - w_off * head2_type_grad_weight * off_grad_v_head2

                # Single sampling after all sequential gradient applications
                ligand_pos_next = pos_model_mean_final + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_final, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)

                # Track predictions from both heads
                # Use head2 selectivity as exp_pred for monitoring (more relevant for selectivity)
                exp_pred = v_on_pred_head2 - v_off_pred_head2  # Selectivity score from head2
                exp_on_traj.append(v_on_pred_head1.clone().cpu())  # Track head1 on-target
                exp_off_traj.append(v_off_pred_head2.clone().cpu())  # Track head2 off-target

            elif guide_mode == 'wo':
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph to save memory
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next_prob = log_sample_categorical(log_model_prob)
                exp_pred = None
            else:
                raise NotImplementedError
            
            ligand_v_next = ligand_v_next_prob.argmax(dim=-1)
            ligand_v = ligand_v_next.detach()  # Break computation graph to save memory
                
            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(ligand_v_next_prob.clone().cpu())   
                
            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
            
            if exp_pred is not None:
                exp_traj.append(exp_pred.clone().cpu())
                # atom_affinity is not available in current model, use empty tensor
                if 'atom_affinity' in preds:
                    exp_atom_traj.append(preds['atom_affinity'].clone().cpu())
                else:
                    # Create dummy atom affinity for compatibility
                    exp_atom_traj.append(torch.zeros_like(exp_pred).cpu())
            else:
                # When exp_pred is None (e.g., on_target_only mode), still add dummy exp_atom_traj
                dummy_affinity = torch.zeros(ligand_v.shape[0], 1, device=ligand_v.device)
                exp_atom_traj.append(dummy_affinity.cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        result = {
            'pos': ligand_pos,
            'v': ligand_v,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }
        
        # Add off-target predictions for selectivity modes
        if guide_mode in ['selectivity', 'sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential', 'head1_head2_sequential']:
            if exp_off_traj:
                result['exp_off_traj'] = exp_off_traj
                result['exp_off'] = exp_off_traj[-1] if exp_off_traj else []
            if exp_on_traj:
                result['exp_on_traj'] = exp_on_traj
                result['exp_on'] = exp_on_traj[-1] if exp_on_traj else []

        return result


    def separate_on_off_gradients(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos,
                                 batch_protein, batch_ligand, off_target_data, t, value_model=None):
        """
        Calculate separate on-target and off-target gradients based on original2.py approach
        Uses only the first off-target (no mean calculation)
        Returns: on_grad1, on_grad2, off_grad1, off_grad2, v_on_pred, v_off_pred
        """
        if value_model is not None:
            value_model.eval()

        with torch.enable_grad():
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob, hard=True, tau=0.5)

            # === ON-TARGET GRADIENT CALCULATION ===
            # Use value_model if provided, otherwise use self
            model_to_use = value_model if value_model is not None else self

            preds_on = model_to_use(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                time_step=t
            )
            v_on_pred = preds_on['final_exp_pred']

            # Calculate on-target gradients (following original2.py approach)
            v_on_pred_log = v_on_pred.log()
            on_grad1 = torch.autograd.grad(v_on_pred, ligand_v_next_prob,
                                         grad_outputs=torch.ones_like(v_on_pred),
                                         retain_graph=True)[0]
            on_grad2 = torch.autograd.grad(v_on_pred_log, ligand_pos,
                                         grad_outputs=torch.ones_like(v_on_pred),
                                         retain_graph=True)[0]

            # === OFF-TARGET GRADIENT CALCULATION (SINGLE OFF-TARGET) ===
            v_off_pred = torch.zeros_like(v_on_pred)
            off_grad1 = torch.zeros_like(on_grad1)
            off_grad2 = torch.zeros_like(on_grad2)

            if off_target_data is not None and len(off_target_data) > 0:
                # Use only the first off-target
                off_data = off_target_data[0]
                try:
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)
                    off_batch_protein = off_data.get('protein_element_batch',
                                                   torch.zeros_like(off_protein_pos[:, 0], dtype=torch.long)).to(protein_pos.device)

                    # Forward pass for off-target
                    preds_off = model_to_use(
                        protein_pos=off_protein_pos,
                        protein_v=off_protein_v,
                        batch_protein=off_batch_protein,
                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v_next,
                        batch_ligand=batch_ligand,
                        time_step=t
                    )
                    v_off_pred = preds_off['final_exp_pred']

                    # Calculate off-target gradients
                    # Since we want to MINIMIZE off-target affinity, we use negative gradients
                    v_off_pred_log = v_off_pred.log()
                    off_grad1 = torch.autograd.grad(-v_off_pred, ligand_v_next_prob,
                                                  grad_outputs=torch.ones_like(v_off_pred),
                                                  retain_graph=True)[0]
                    off_grad2 = torch.autograd.grad(-v_off_pred_log, ligand_pos,
                                                  grad_outputs=torch.ones_like(v_off_pred),
                                                  retain_graph=False)[0]

                except Exception as e:
                    print(f"Error processing off-target: {e}")
                    v_off_pred = torch.zeros_like(v_on_pred)
                    off_grad1 = torch.zeros_like(on_grad1)
                    off_grad2 = torch.zeros_like(on_grad2)

            return on_grad1, on_grad2, off_grad1, off_grad2, v_on_pred, v_off_pred

    def joint_separate_on_off_gradients(self, ligand_v_next_prob, ligand_pos, protein_v, protein_pos,
                                       batch_protein, batch_ligand, off_target_data, t):
        """
        Calculate separate on-target and off-target gradients using joint model (self)
        Uses only the first off-target (no mean calculation)
        """
        with torch.enable_grad():
            ligand_v_next_prob = ligand_v_next_prob.detach().requires_grad_(True)
            ligand_pos = ligand_pos.detach().requires_grad_(True)
            ligand_v_next = F.gumbel_softmax(ligand_v_next_prob, hard=True, tau=0.5)

            # === ON-TARGET GRADIENT CALCULATION ===
            time_step_on = torch.zeros(1, dtype=torch.long, device=protein_pos.device)
            protein_id_on = torch.zeros_like(batch_protein)

            preds_on = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v_next,
                batch_ligand=batch_ligand,
                protein_id=protein_id_on,
                time_step=time_step_on
            )

            v_on_pred = preds_on.get('v_on_pred', preds_on.get('final_exp_pred'))

            # Calculate on-target gradients (following original2.py approach)
            v_on_pred_log = v_on_pred.log()
            on_grad1 = torch.autograd.grad(v_on_pred, ligand_v_next_prob,
                                         grad_outputs=torch.ones_like(v_on_pred),
                                         retain_graph=True)[0]
            on_grad2 = torch.autograd.grad(v_on_pred_log, ligand_pos,
                                         grad_outputs=torch.ones_like(v_on_pred),
                                         retain_graph=True)[0]

            # === OFF-TARGET GRADIENT CALCULATION (SINGLE OFF-TARGET) ===
            v_off_pred = torch.zeros_like(v_on_pred)
            off_grad1 = torch.zeros_like(on_grad1)
            off_grad2 = torch.zeros_like(on_grad2)

            if off_target_data is not None and len(off_target_data) > 0:
                # Use only the first off-target
                off_data = off_target_data[0]
                try:
                    off_protein_v = off_data['protein_atom_feature'].float().to(protein_pos.device)
                    off_protein_pos = off_data['protein_pos'].to(protein_pos.device)
                    off_batch_protein = off_data.get('protein_element_batch',
                                                   torch.zeros_like(off_protein_pos[:, 0], dtype=torch.long)).to(protein_pos.device)

                    off_protein_id = torch.zeros_like(off_batch_protein)
                    time_step_off = torch.zeros(1, dtype=torch.long, device=protein_pos.device)

                    # Forward pass for off-target
                    preds_off = self(
                        protein_pos=off_protein_pos,
                        protein_v=off_protein_v,
                        batch_protein=off_batch_protein,
                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v_next,
                        batch_ligand=batch_ligand,
                        protein_id=off_protein_id,
                        time_step=time_step_off
                    )

                    v_off_pred = preds_off.get('v_off_pred', preds_off.get('final_exp_pred'))

                    # Calculate off-target gradients (negative to minimize off-target affinity)
                    v_off_pred_log = v_off_pred.log()
                    off_grad1 = torch.autograd.grad(-v_off_pred, ligand_v_next_prob,
                                                  grad_outputs=torch.ones_like(v_off_pred),
                                                  retain_graph=True)[0]
                    off_grad2 = torch.autograd.grad(-v_off_pred_log, ligand_pos,
                                                  grad_outputs=torch.ones_like(v_off_pred),
                                                  retain_graph=False)[0]

                except Exception as e:
                    print(f"Error processing off-target: {e}")
                    v_off_pred = torch.zeros_like(v_on_pred)
                    off_grad1 = torch.zeros_like(on_grad1)
                    off_grad2 = torch.zeros_like(on_grad2)

            return on_grad1, on_grad2, off_grad1, off_grad2, v_on_pred, v_off_pred

    def forward_separate_ba(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                           time_step=None, return_all=False, fix_x=False):
        """
        Separate BA Forward: Process protein and ligand separately for BA prediction
        - Generation: protein-ligand complex graph (with interactions)
        - BA prediction: separate protein and ligand graphs (no interactions)
        """
        batch_size = batch_protein.max().item() + 1

        # Process ligand features
        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        elif len(init_ligand_v.shape) == 2:
            pass
        else:
            raise ValueError("Invalid ligand feature shape")

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        # Atom embeddings
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        # Node indicator
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # === 1. Complex graph for generation (with interactions) ===
        h_complex, pos_complex, batch_complex, mask_ligand_complex, mask_protein_complex, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            protein_id=None
        )

        # Process through UniTransformer with interactions
        outputs_complex = self.refine_net(h_complex, pos_complex, mask_ligand_complex, batch_complex,
                                         protein_id=None, return_all=False, fix_x=fix_x)

        # Extract ligand results for generation (from interaction graph)
        final_ligand_h = outputs_complex['h'][mask_ligand_complex]
        final_ligand_pos = outputs_complex['x'][mask_ligand_complex]
        final_ligand_v = self.v_inference(final_ligand_h)

        # === 2. Separate graphs for BA prediction (no interactions) ===
        # Process protein separately (no ligand interaction)
        mask_protein_only = torch.zeros(len(h_protein), dtype=torch.bool, device=h_protein.device)
        outputs_protein_only = self.refine_net(h_protein, protein_pos, mask_protein_only, batch_protein,
                                              protein_id=None, return_all=False, fix_x=fix_x)

        # Process ligand separately (no protein interaction)
        mask_ligand_only = torch.ones(len(init_ligand_h), dtype=torch.bool, device=init_ligand_h.device)
        outputs_ligand_only = self.refine_net(init_ligand_h, init_ligand_pos, mask_ligand_only, batch_ligand,
                                             protein_id=None, return_all=False, fix_x=fix_x)

        # Get separate embeddings for BA prediction
        h_mol_ligand_separate = scatter_mean(outputs_ligand_only['h'], batch_ligand, dim=0)  # [batch_size, hidden_dim]
        h_mol_protein_separate = scatter_mean(outputs_protein_only['h'], batch_protein, dim=0)  # [batch_size, hidden_dim]

        # === 3. Single-head BA prediction ===
        # Concatenate separate protein and ligand embeddings
        separate_ba_input = torch.cat([h_mol_ligand_separate, h_mol_protein_separate], dim=1)
        v_separate_ba_pred = self.separate_ba_head(separate_ba_input).squeeze(-1)

        # Return results (use complex graph results for generation)
        preds = {
            'ligand_pos': final_ligand_pos,
            'ligand_v': final_ligand_v,
            'ligand_h': final_ligand_h,
            'protein_h': outputs_complex['h'],
            'v_separate_ba_pred': v_separate_ba_pred,
            'batch_ligand': batch_ligand
        }

        if return_all:
            preds.update({
                'h_mol_ligand_separate': h_mol_ligand_separate,
                'h_mol_protein_separate': h_mol_protein_separate,
            })

        return preds

    def get_diffusion_loss_separate_ba(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
            affinity, time_step=None
    ):
        """
        Separate BA diffusion loss:
        - Generation is trained with protein-ligand interactions
        - BA prediction is trained without protein-ligand interactions (separate processing)
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode, center_ligand=self.center_ligand)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN using separate BA architecture
        preds = self.forward_separate_ba(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['ligand_pos'], preds['ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # atom position loss
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # Separate BA loss (non-interaction based)
        affinity_normalized = affinity

        # Single-head BA prediction loss
        loss_separate_ba = F.mse_loss(preds['v_separate_ba_pred'].squeeze(), affinity_normalized.squeeze())

        # Combined affinity loss
        loss_exp = loss_separate_ba

        # Total loss
        loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss_separate_ba': loss_separate_ba,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_separate_ba': preds['v_separate_ba_pred'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['ligand_h']
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

# %%
