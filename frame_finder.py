from lyc.utils import get_tokenizer, get_model
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, processor, get_dataloader
from lyc.train import get_base_hf_args, HfTrainer
from lyc.eval import tagging_eval_for_trainer
import sys
import numpy as np
import datasets
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer
from transformers.integrations import TensorBoardCallback
import torch

def combine_func(df):
    """combine a dataframe group to a one-line instance.

    Args:
        df ([dataframe]): a dataframe, represents a group

    Returns:
        a one-line dict
    """
    label_values = np.stack(df['frame_tags'].values)
    processed = np.zeros_like(label_values)

    for token_id in range(label_values.shape[1]):
        labels = label_values[:, token_id]
        for i in range(len(labels)):
            if labels[i] != 0:
                processed[i, token_id] = labels[i]
                break

    aggregated_tags = processed.sum(axis=0)
    result = df.iloc[0].to_dict()
    result['frame_tags'] = aggregated_tags

    return result

def tokenize_alingn_labels_replace_with_mask_and_add_type_ids(ds):
    results={}

    target_index = None
    for i in range(len(ds['frame_tags'])):
        if ds['frame_tags'][i]:
            target_index = i
    tokens = ds['tokens']
    tokens[target_index] = '<mask>'
    ds['tokens'] = tokens

    for k,v in ds.items():
        if 'id' in k:
            results[k]=v
            continue
        if 'tag' not in k:
            out_=tokenizer(v, is_split_into_words=True)
            results.update(out_)
    labels={}
    for i, column in enumerate([k for k in ds.keys() if 'tag' in k]):
        label = ds[column]
        words_ids = out_.word_ids()
        previous_word_idx = None
        label_ids = []
        is_target = []
        for word_idx in words_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx!=previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            if word_idx == target_index:
                is_target.append(1)
            else:
                is_target.append(0)
            previous_word_idx = word_idx
        labels[column] = label_ids
        labels['is_target'] = is_target
    
    results.update(labels)
    return results

if __name__ == '__main__':
    model_name, data_dir = sys.argv[1:]

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)
    script = get_hf_ds_scripts_path('sesame')
    ds = datasets.load_dataset(script, data_dir=data_dir)

    label_list = ds['train'].features['frame_tags'].feature.names
    # for k,v in ds.items():
    #     ds[k] = processor.combine(v, 'sent_id', combine_func)

    # processor.tokenizer = tokenizer
    # ds = ds.map(
    #     processor._tokenize_and_alingn_labels
    # )

    ds = ds.map(
        tokenize_alingn_labels_replace_with_mask_and_add_type_ids
    )

    train_ds = datasets.concatenate_datasets([ds['train'], ds['test']])
    train_ds = train_ds.rename_column('frame_tags', 'labels')
    train_ds = train_ds.rename_column('is_target', 'token_type_ids')

    eval_ds = ds['validation']
    eval_ds = eval_ds.rename_column('frame_tags', 'labels')
    eval_ds = eval_ds.rename_column('is_target', 'token_type_ids')

    args = get_base_hf_args(
        output_dir='checkpoints/frame_finder/',
        train_batch_size=24,
        epochs=3,
        lr=5e-5,
        logging_steps = 1
    )

    model = get_model(RobertaForTokenClassification, model_name, num_labels = len(label_list))
    model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, 768)
    model._init_weights(model.roberta.embeddings.token_type_embeddings)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],
        compute_metrics=tagging_eval_for_trainer
    )

    trainer.train()
    trainer.save_model()

    result = trainer.evaluate()
    print(result)