import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import os

from modeling_gpt_bert_test import GPTBERTForMaskedLM, GPTBERTForCausalLM
from configuration_gpt_bert import ModelConfig

def fix_broken_keys_and_load(model_path, state_dict_name="pytorch_model.bin", causal_lm=True):
    """Load a legacy local checkpoint or a packaged Hugging Face Hub model.

    Local directories retain the original key-repair behavior when the old
    state dict is present. Hub model IDs load through the custom AutoClass
    metadata added by ``upload_models_to_hub.py``.
    """
    if os.path.isdir(model_path):
        state_dict_path = os.path.join(model_path, state_dict_name)
        if os.path.isfile(state_dict_path):
            new_state_dict_path = os.path.join(model_path, "pytorch_model.bin")
            sd = torch.load(state_dict_path, map_location="cpu")
            fixed = {}
            for k, v in sd.items():
                if k.startswith("transformer."):
                    k = "model." + k[len("transformer."):]
                elif k.startswith("classifier."):
                    k = k.replace("classifier.", "lm_head.", 1)
                elif k.startswith("embedding."):
                    k = "model.embedding." + k[len("embedding."):]
                fixed[k] = v
            torch.save(fixed, new_state_dict_path)

        config = ModelConfig.from_pretrained(model_path)
        model_class = GPTBERTForCausalLM if causal_lm else GPTBERTForMaskedLM
        model, info = model_class.from_pretrained(
            model_path, config=config, output_loading_info=True
        )
    else:
        auto_class = AutoModelForCausalLM if causal_lm else AutoModelForMaskedLM
        model, info = auto_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            output_loading_info=True,
        )

    print("missing keys:", info["missing_keys"])
    print("unexpected keys:", info["unexpected_keys"])
    print("mismatched keys:", info.get("mismatched_keys", []))
    print("error msgs:", info["error_msgs"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model.eval()

    return model, tokenizer