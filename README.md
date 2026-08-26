Modified training / loading script for GPT-BERT. See original repo for details. 

## Training

For the most part, follow the instructions in the [original GPT-BERT repository](https://github.com/ltgoslo/gpt-bert).

1. Upload the raw training and development data to the `data/` directory.
2. Train the tokenizer.
3. Use the trained tokenizer to tokenize the training and development data.
4. Run [`pretraining/train_single_gpu.py`](pretraining/train_single_gpu.py) with the tokenized training and development data (the .bin files).

   This implementation differs from the original model, which used distributed training. If you are training for ~10 epochs, make sure the optimizer is set to AdamW.

5. Find the final model in `model_checkpoints/`.

> [!NOTE]
> Intermediate checkpoints are not saved; only the final model is saved.

> [!NOTE]
> A few hyperparameters necessarily differ from the original implementation and were not selected systematically. In experiments, these hyperparameters produced comparable BLiMP performance to BabyLM baselines (approximately 0.8 F1 after 10 epochs).

## Loading

Place the following files together in a single directory. This directory will be the model path:

- `pytorch_model.bin`
- `tokenizer.json`
- `config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`

Once these files are in the model directory, you should be able to load the model using the `GPTBertForCausalLM` and `GPTBertForMaskedLM` classes defined in `modeling_gpt_bert_test.py`, along with the configuration in `configuration_gpt_bert.py`. These implementations are largely the same as those included in the public [BabyLM GPT-BERT model releases](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus).

### Loading issues

Two issues were encountered while loading trained models:

1. **Unexpected output format with minicons.** Setting `return_dict = True` did not return the format expected by minicons. The modified `modeling_gpt_bert.py` in this repository should fix this. If you use the public Hugging Face releases, such as the [BabyLM GPT-BERT causal-focus model](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus), you may run into this issue.
2. **Differences in model component names.** Some model components may have different names, likely because of dependency version differences. The helper in [`loading_utils.py`](loading_utils.py) works around this by renaming model components as needed. The exact behavior may vary across environments and versions.

## Evaluating

The included [`predict_blimp.py`](predict_blimp.py) provides an example of how to load and evaluate a model on BLiMP using minicons causalLM scorer. While the code technically will load a masked LM scorer, because of how GPT-BERT does MNTP instead of pure MLM, the minicons Masked scorer won't work as intended. 

## Original GPT-BERT citation
```bibtex
@inproceedings{charpentier-samuel-2024-bert,
    title = "{BERT} or {GPT}: why not both?",
    author = "Charpentier, Lucas Georges Gabriel  and
      Samuel, David",
    editor = "Hu, Michael Y.  and
      Mueller, Aaron  and
      Ross, Candace  and
      Williams, Adina  and
      Linzen, Tal  and
      Zhuang, Chengxu  and
      Choshen, Leshem  and
      Cotterell, Ryan  and
      Warstadt, Alex  and
      Wilcox, Ethan Gotlieb",
    booktitle = "The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-babylm.24/",
    pages = "262--283",
}
```
