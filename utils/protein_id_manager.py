"""
Protein ID Manager - 통합 단백질 ID 관리 시스템

이 모듈은 validation set의 단백질 중복 문제를 해결하고,
모든 스크립트에서 일관된 단백질 ID 매핑을 제공합니다.
"""

import json
import collections
from typing import Dict, List, Tuple, Set
import warnings

class ProteinIDManager:
    def __init__(self, validation_info_path: str = './scratch2/data/multipro_validation_info.json'):
        """
        단백질 ID 관리자 초기화

        Args:
            validation_info_path: validation info JSON 파일 경로
        """
        self.validation_info_path = validation_info_path
        self.validation_info = None
        self.protein_to_validation_ids = {}  # protein_name -> [validation_id1, validation_id2, ...]
        self.validation_id_to_protein = {}   # validation_id -> protein_name
        self.unique_proteins = []            # 중복 제거된 고유 단백질 리스트
        self.unique_protein_to_representative_id = {}  # protein_name -> 대표 validation_id

        self._load_and_analyze()

    def _load_and_analyze(self):
        """validation info를 로드하고 중복 분석"""
        with open(self.validation_info_path, 'r') as f:
            self.validation_info = json.load(f)

        # 매핑 구축
        for i, entry in enumerate(self.validation_info):
            protein_name = entry['protein_dir']
            self.validation_id_to_protein[i] = protein_name

            if protein_name not in self.protein_to_validation_ids:
                self.protein_to_validation_ids[protein_name] = []
            self.protein_to_validation_ids[protein_name].append(i)

        # 고유 단백질 리스트 생성 (대표 ID는 가장 작은 validation_id 사용)
        self.unique_proteins = list(self.protein_to_validation_ids.keys())
        for protein_name in self.unique_proteins:
            validation_ids = self.protein_to_validation_ids[protein_name]
            representative_id = min(validation_ids)  # 가장 작은 ID를 대표로 사용
            self.unique_protein_to_representative_id[protein_name] = representative_id

        print(f"[ProteinIDManager] Loaded {len(self.validation_info)} validation entries")
        print(f"[ProteinIDManager] Found {len(self.unique_proteins)} unique proteins")

        # 중복 정보 출력
        duplicates = {name: ids for name, ids in self.protein_to_validation_ids.items() if len(ids) > 1}
        if duplicates:
            print(f"[ProteinIDManager] WARNING: {len(duplicates)} proteins have duplicates:")
            for name, ids in list(duplicates.items())[:5]:  # 처음 5개만 출력
                print(f"  {name}: validation_ids {ids}")
            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more")

    def get_protein_name(self, validation_id: int) -> str:
        """validation_id로부터 단백질 이름 반환"""
        return self.validation_id_to_protein.get(validation_id, f"unknown_{validation_id}")

    def get_all_validation_ids_for_protein(self, protein_name: str) -> List[int]:
        """단백질 이름에 대한 모든 validation_id 반환"""
        return self.protein_to_validation_ids.get(protein_name, [])

    def get_representative_id(self, protein_name: str) -> int:
        """단백질 이름의 대표 validation_id 반환"""
        return self.unique_protein_to_representative_id.get(protein_name, -1)

    def validate_id_selection(self, on_target_id: int, off_target_ids: List[int]) -> Tuple[bool, List[str]]:
        """
        on-target과 off-target ID 선택의 유효성 검사

        Args:
            on_target_id: On-target validation ID
            off_target_ids: Off-target validation IDs 리스트

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # On-target 단백질 이름
        on_target_protein = self.get_protein_name(on_target_id)

        # 중복 검사
        all_ids = [on_target_id] + off_target_ids
        used_proteins = set()

        for val_id in all_ids:
            protein_name = self.get_protein_name(val_id)

            if protein_name in used_proteins:
                errors.append(f"Protein '{protein_name}' is used multiple times")
            else:
                used_proteins.add(protein_name)

        # Off-target ID 유효성 검사
        for off_id in off_target_ids:
            if off_id == on_target_id:
                errors.append(f"Off-target ID {off_id} is same as on-target ID")

            off_protein = self.get_protein_name(off_id)
            if off_protein == on_target_protein:
                errors.append(f"Off-target ID {off_id} ({off_protein}) is same protein as on-target ID {on_target_id}")

        return len(errors) == 0, errors

    def get_safe_off_target_ids(self, on_target_id: int, requested_off_target_ids: List[int]) -> List[int]:
        """
        안전한 off-target ID 리스트 반환 (중복 단백질 자동 교체)

        Args:
            on_target_id: On-target validation ID
            requested_off_target_ids: 요청된 off-target IDs

        Returns:
            중복이 제거된 안전한 off-target ID 리스트
        """
        on_target_protein = self.get_protein_name(on_target_id)
        safe_off_target_ids = []
        used_proteins = {on_target_protein}

        for off_id in requested_off_target_ids:
            off_protein = self.get_protein_name(off_id)

            if off_protein not in used_proteins:
                # 안전한 ID
                safe_off_target_ids.append(off_id)
                used_proteins.add(off_protein)
            else:
                # 중복된 단백질 - 대체 ID 찾기
                print(f"[ProteinIDManager] WARNING: Off-target ID {off_id} ({off_protein}) conflicts with used protein")
                alternative_id = self._find_alternative_protein_id(used_proteins)
                if alternative_id is not None:
                    alternative_protein = self.get_protein_name(alternative_id)
                    print(f"[ProteinIDManager] Replacing with ID {alternative_id} ({alternative_protein})")
                    safe_off_target_ids.append(alternative_id)
                    used_proteins.add(alternative_protein)
                else:
                    print(f"[ProteinIDManager] WARNING: Could not find alternative for {off_protein}")

        return safe_off_target_ids

    def _find_alternative_protein_id(self, used_proteins: Set[str]) -> int:
        """사용되지 않은 단백질의 대표 ID 찾기"""
        for protein_name in self.unique_proteins:
            if protein_name not in used_proteins:
                return self.get_representative_id(protein_name)
        return None

    def get_protein_summary(self) -> Dict:
        """단백질 정보 요약 반환"""
        duplicates = {name: ids for name, ids in self.protein_to_validation_ids.items() if len(ids) > 1}

        return {
            'total_validation_entries': len(self.validation_info),
            'unique_proteins': len(self.unique_proteins),
            'duplicate_proteins': len(duplicates),
            'duplicate_details': duplicates
        }

    def print_summary(self):
        """단백질 정보 요약 출력"""
        summary = self.get_protein_summary()
        print(f"=== Protein ID Manager Summary ===")
        print(f"Total validation entries: {summary['total_validation_entries']}")
        print(f"Unique proteins: {summary['unique_proteins']}")
        print(f"Proteins with duplicates: {summary['duplicate_proteins']}")

        if summary['duplicate_proteins'] > 0:
            print(f"\nDuplicate proteins (first 5):")
            for name, ids in list(summary['duplicate_details'].items())[:5]:
                print(f"  {name}: {ids}")


# 전역 인스턴스 생성 (싱글톤 패턴)
_protein_manager = None

def get_protein_manager() -> ProteinIDManager:
    """전역 ProteinIDManager 인스턴스 반환"""
    global _protein_manager
    if _protein_manager is None:
        _protein_manager = ProteinIDManager()
    return _protein_manager

# 편의 함수들
def validate_protein_ids(on_target_id: int, off_target_ids: List[int]) -> Tuple[bool, List[str]]:
    """단백질 ID 유효성 검사 (편의 함수)"""
    return get_protein_manager().validate_id_selection(on_target_id, off_target_ids)

def get_safe_protein_ids(on_target_id: int, off_target_ids: List[int]) -> Tuple[int, List[int]]:
    """안전한 단백질 ID 조합 반환 (편의 함수)"""
    manager = get_protein_manager()
    safe_off_targets = manager.get_safe_off_target_ids(on_target_id, off_target_ids)
    return on_target_id, safe_off_targets

def get_protein_name_by_id(validation_id: int) -> str:
    """validation_id로 단백질 이름 반환 (편의 함수)"""
    return get_protein_manager().get_protein_name(validation_id)


if __name__ == "__main__":
    # 테스트
    manager = ProteinIDManager()
    manager.print_summary()

    # 중복 테스트
    print("\n=== Testing Duplicate Detection ===")
    on_target = 0  # BACE1_HUMAN_49_451_0
    off_targets = [12, 62]  # Both are also BACE1_HUMAN_49_451_0

    is_valid, errors = manager.validate_id_selection(on_target, off_targets)
    print(f"Validation result: {is_valid}")
    for error in errors:
        print(f"ERROR: {error}")

    # 안전한 ID 생성 테스트
    print(f"\n=== Testing Safe ID Generation ===")
    safe_on, safe_off = get_safe_protein_ids(on_target, off_targets)
    print(f"Original: on_target={on_target}, off_targets={off_targets}")
    print(f"Safe: on_target={safe_on}, off_targets={safe_off}")

    print(f"On-target protein: {get_protein_name_by_id(safe_on)}")
    for i, off_id in enumerate(safe_off):
        print(f"Off-target {i}: ID {off_id} ({get_protein_name_by_id(off_id)})")