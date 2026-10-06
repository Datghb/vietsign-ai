"""External real-data source registry."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataSource(BaseModel):
    name: str
    task: str
    source_url: str
    access_type: str
    license: str = "unknown"
    expected_format: str
    local_target_directory: Path
    status: str


class DataSourceRegistry(BaseModel):
    sources: list[DataSource] = Field(default_factory=list)

    def get(self, name: str) -> DataSource:
        normalized = _normalize_name(name)
        for source in self.sources:
            if _normalize_name(source.name) == normalized:
                return source
        raise KeyError(f"Unknown data source: {name}")


def load_source_registry(path: str | Path = "configs/data/sources.yaml") -> DataSourceRegistry:
    with Path(path).open("r", encoding="utf-8") as file:
        return DataSourceRegistry.model_validate(yaml.safe_load(file) or {})


def source_status_matrix(registry: DataSourceRegistry | None = None) -> list[dict[str, Any]]:
    registry = registry or load_source_registry()
    rows = []
    for source in registry.sources:
        local = Path(source.local_target_directory)
        rows.append(
            {
                "dataset": source.name,
                "task": source.task,
                "real": True,
                "source_url": source.source_url,
                "license_access": f"{source.license} / {source.access_type}",
                "local_target_directory": str(local),
                "local_status": "present" if local.exists() and any(local.iterdir()) else "missing",
                "registry_status": source.status,
                **_capability_flags(source.name),
            }
        )
    return rows


def fetch_huggingface_metadata(repo_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo_id}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _capability_flags(name: str) -> dict[str, bool | str]:
    key = _normalize_name(name)
    if key == "voya-vsl":
        return {
            "video": False,
            "keypoints": True,
            "isolated_labels": True,
            "gloss_sequence": False,
            "Vietnamese_text": False,
            "signer_ids": False,
            "usable_for_ISLR": True,
            "usable_for_CSLR": False,
            "usable_for_SLT": False,
        }
    if key == "vsl400":
        return {
            "video": True,
            "keypoints": False,
            "isolated_labels": True,
            "gloss_sequence": False,
            "Vietnamese_text": False,
            "signer_ids": True,
            "usable_for_ISLR": "pending_authorized_access",
            "usable_for_CSLR": False,
            "usable_for_SLT": False,
        }
    if key == "parallel-corpus-vie-vsl":
        return {
            "video": False,
            "keypoints": False,
            "isolated_labels": False,
            "gloss_sequence": True,
            "Vietnamese_text": True,
            "signer_ids": False,
            "usable_for_ISLR": False,
            "usable_for_CSLR": False,
            "usable_for_SLT": True,
        }
    if key == "visl-news":
        return {
            "video": "metadata_only_until_media_resolved",
            "keypoints": False,
            "isolated_labels": False,
            "gloss_sequence": False,
            "Vietnamese_text": True,
            "signer_ids": True,
            "usable_for_ISLR": False,
            "usable_for_CSLR": False,
            "usable_for_SLT": "gloss_free_sentence_level_candidate",
        }
    return {}


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")
