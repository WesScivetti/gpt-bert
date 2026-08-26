"""
Minor modifications to original: https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus/blob/main/configuration_gpt_bert.py
"""
from __future__ import annotations

import json
import pathlib
import copy

from typing import Any
from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):
    # Unique identifier used by Hugging Face AutoConfig when this custom
    # architecture is loaded from the Hub with trust_remote_code=True.
    model_type = "gpt_bert"

    def __init__(self, config_file=None, **kwargs):
        super().__init__(**kwargs)

        self.attention_probs_dropout_prob = kwargs.get("attention_probs_dropout_prob", 0.1)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.intermediate_size = kwargs.get("intermediate_size", 2560)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 512)
        self.max_sequence_length = kwargs.get("max_sequence_length", self.max_position_embeddings)
        self.position_bucket_size = kwargs.get("position_bucket_size", 32)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_layers = kwargs.get("num_layers", self.num_hidden_layers)
        self.vocab_size = kwargs.get("vocab_size", 16384)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-7)

        if config_file is not None:
            import json, pathlib
            if isinstance(config_file, str):
                config_file = pathlib.Path(config_file)
            config = json.load(config_file.open("r"))
            for key, value in config.items():
                setattr(self, key, value)

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        output: dict[str, Any] = copy.deepcopy(self.__dict__)
        return output

    # def to_json_string(self) -> str:
    #     """Serializes this instance to a JSON string."""
    #     return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: pathlib.Path | str) -> None:
        """Save this instance to a json file."""
        if isinstance(json_file_path, str):
            json_file_path: pathlib.Path = pathlib.Path(json_file_path)
        with json_file_path.open("w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())
