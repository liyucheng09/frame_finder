from lyc.utils import get_tokenizer, get_model
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, processor, get_dataloader
from lyc.train import get_base_hf_args, HfTrainer
import sys
import numpy as np
import datasets
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification
from transformers.integrations import TensorBoardCallback

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


if __name__ == '__main__':
    model_name, data_dir = sys.argv[1:]

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)
    script = get_hf_ds_scripts_path('sesame')
    ds = datasets.load_dataset(script, data_dir=data_dir)

    label_list = ds['train'].features['frame_tags'].feature.names
    for k,v in ds.items():
        ds[k] = processor.combine(v, 'sent_id', combine_func)
    processor.tokenizer = tokenizer
    ds = ds.map(
        processor._tokenize_and_alingn_labels
    )

    train_ds = datasets.concatenate_datasets([ds['train'], ds['test']])
    train_ds = train_ds.rename_column('frame_tags', 'labels')

    eval_ds = ds['validation']
    eval_ds = eval_ds.rename_column('frame_tags', 'labels')

    args = get_base_hf_args(
        output_dir='checkpoints/frame_finder/',
        train_batch_size=24,
        epochs=3,
        lr=5e-5,
        logging_steps = 1
    )

    model = get_model(RobertaForTokenClassification, model_name, num_labels = len(label_list))
    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)

    trainer = HfTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],
    )

    trainer.train()
    trainer.save_model()

    result = trainer.evaluate()
    print(result)