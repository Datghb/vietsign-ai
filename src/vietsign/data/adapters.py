"""Dataset adapters that map source-specific metadata into SignSample records."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from vietsign.data.config import DatasetConfig
from vietsign.data.schema import SignSample


class DatasetLoadError(RuntimeError):
    """Raised when a dataset source cannot be loaded."""


class BaseDatasetAdapter(ABC):
    """Common dataset adapter interface."""

    def __init__(self, name: str, root: str | Path) -> None:
        self.name = name
        self.root = Path(root)

    @abstractmethod
    def load_samples(self) -> list[SignSample]:
        """Load normalized samples."""


class CustomDatasetAdapter(BaseDatasetAdapter):
    """Load a normalized custom CSV manifest."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "sample_id",
        "video_path",
        "label",
        "label_id",
        "split",
    }

    def __init__(
        self,
        name: str,
        root: str | Path,
        metadata_csv: str | Path,
    ) -> None:
        super().__init__(name=name, root=root)
        self.metadata_csv = Path(metadata_csv)

    def load_samples(self) -> list[SignSample]:
        if not self.metadata_csv.exists():
            raise DatasetLoadError(f"Metadata CSV not found: {self.metadata_csv}")

        with self.metadata_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = self.REQUIRED_COLUMNS - fieldnames
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise DatasetLoadError(f"Metadata CSV missing required columns: {missing}")
            return list(self._rows_to_samples(reader))

    def _rows_to_samples(self, rows: Iterable[dict[str, str]]) -> Iterable[SignSample]:
        for line_number, row in enumerate(rows, start=2):
            try:
                yield SignSample(
                    sample_id=row["sample_id"],
                    video_path=self._resolve_video_path(row["video_path"]),
                    label=row["label"],
                    label_id=int(row["label_id"]),
                    signer_id=row.get("signer_id") or None,
                    split=row["split"],
                    dataset=row.get("dataset") or self.name,
                    camera=row.get("camera") or None,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "sample_id",
                            "video_path",
                            "label",
                            "label_id",
                            "signer_id",
                            "split",
                            "dataset",
                            "camera",
                        }
                        and value not in (None, "")
                    },
                )
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                raise DatasetLoadError(
                    f"Invalid metadata row {line_number} in {self.metadata_csv}: {error}"
                ) from error

    def _resolve_video_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root / path


class VSL400Adapter(CustomDatasetAdapter):
    """Conservative VSL-400 adapter using normalized metadata."""

    def load_samples(self) -> list[SignSample]:
        if not self.root.exists():
            raise DatasetLoadError(
                "VSL-400 root directory not found. Expected the dataset at "
                f"{self.root}. Update configs/data/vsl400.yaml if your dataset lives elsewhere."
            )
        if not self.metadata_csv.exists():
            raise DatasetLoadError(
                "VSL-400 native directory parsing has not been verified yet. "
                f"Provide a normalized metadata CSV at {self.metadata_csv} or update the config."
            )
        return super().load_samples()


class VSL1000Adapter(BaseDatasetAdapter):
    """Reserved adapter for VSL-1000 once its real structure is verified."""

    def load_samples(self) -> list[SignSample]:
        raise DatasetLoadError(
            "VSL-1000 adapter is reserved, but its directory structure has not been verified."
        )


def create_adapter(config: DatasetConfig) -> BaseDatasetAdapter:
    """Create a dataset adapter from config."""

    source = config.dataset
    if source.adapter == "custom_csv":
        if source.metadata_csv is None:
            raise DatasetLoadError("custom_csv adapter requires dataset.metadata_csv")
        return CustomDatasetAdapter(source.name, source.root, source.metadata_csv)
    if source.adapter == "vsl400":
        metadata_csv = source.metadata_csv or Path("data/metadata/vsl400.csv")
        return VSL400Adapter(source.name, source.root, metadata_csv)
    if source.adapter == "vsl1000":
        return VSL1000Adapter(source.name, source.root)
    raise DatasetLoadError(f"Unsupported dataset adapter: {source.adapter}")
