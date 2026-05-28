from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_path: Path
    artifacts_dir: Path

    @staticmethod
    def default() -> "Paths":
        root = Path(__file__).resolve().parents[1]
        return Paths(
            root=root,
            data_path=root / "marketing_and_sales.csv",
            artifacts_dir=root / "artifacts",
        )


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
