from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, get_dataloader
from lyc.utils import get_model, get_tokenizer
from transformers import RobertaForTokenClassification, Trainer, DataCollatorForTokenClassification
import sys
from tqdm import tqdm
import torch
import json
import datasets
import os
import numpy as np

def tokenize_alingn_labels_replace_with_mask_and_add_type_ids(ds):
    results={}

    target_index = ds['word_index']
    tokens = ds['tokens']
    results['masked_word'] = tokens[target_index]
    tokens[target_index] = '<mask>'
    ds['tokens'] = tokens

    for k,v in ds.items():
        if k != 'tokens':
            continue
        else:
            out_=tokenizer(v, is_split_into_words=True)
            results.update(out_)

    words_ids = out_.word_ids()
    label_sequence = [0 for i in range(len(words_ids))]
    target_mask = [0 for i in range(len(words_ids))]
    word_idx = words_ids.index(target_index)

    label_sequence[word_idx] = ds['label']
    target_mask[word_idx] = 1

    results['target_mask'] = target_mask
    results['labels'] = label_sequence
    results['tokenized_taregt_word_index'] = word_idx
    results['token_level_label'] = ds['label']
    return results

if __name__ == '__main__':

    model_name, data_dir, = sys.argv[1:]
    save_folder = ''
    output_dir = os.path.join(save_folder, 'checkpoints/roberta_seq/')
    prediction_output_file = os.path.join(output_dir, 'prediction_output.csv')

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)
    script_path = get_hf_ds_scripts_path('vua20')

    data_files={'train': os.path.join(data_dir, 'train.tsv'), 'test': os.path.join(data_dir, 'test.tsv')}
    ds = datasets.load_dataset(script_path, data_files=data_files)
    ds = ds.map(tokenize_alingn_labels_replace_with_mask_and_add_type_ids)
    ds.remove_columns_('label')
    ds.rename_column_('target_mask', 'token_type_ids')

    # ds = get_tokenized_ds(script_path, tokenizer, max_length=128, tokenize_func='general', tokenize_cols=['tokens'],
    #     is_split_into_words=True, return_word_ids=True,
    #     data_files={
    #         'train':'/Users/liyucheng/projects/acl2021-metaphor-generation-conceptual-main/EM/data/VUA20/train.tsv',
    #         'test': '/Users/liyucheng/projects/acl2021-metaphor-generation-conceptual-main/EM/data/VUA20/test.tsv'})
    
    # dl = get_dataloader(ds['test'], batch_size=8, cols=['input_ids', 'attention_mask', 'word_index', 'words_ids', 'label'])
    
    model = get_model(RobertaForTokenClassification, model_name)
    # model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, 768)

    with open('frame_labels.json', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {v:k for k,v in label2id.items()}

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)
    trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer)
    
    pred_out = trainer.predict(ds)
    pred = pred_out.predictions
    df = ds.to_pandas()
    df['frames'] = np.argmax(pred, axis=-1)[np.arange(len(df.index)), df['tokenized_taregt_word_index'].values]
    df['frames'] = df['frames'].apply(lambda x:id2label[x])

    file = open(prediction_output_file, 'w', encoding='utf-8')
    for sent_id, group in df.groupby('sent_id'):
        tokens = group.iloc[0]['tokens']
        target_ids = group['word_index'].values
        labels = group['token_level_label'].values
        frames = group['frames'].values
        words = group['masked_word'].values
        id2label_and_frame = {idx:[label, frame, word] for idx, label, frame, word in zip(target_ids, labels, frames, words)}
        for w_idx, word in enumerate(tokens):
            if w_idx in target_ids:
                is_target = 1
                label = id2label_and_frame[w_idx][0]
                frame = id2label_and_frame[w_idx][1]
                word = id2label_and_frame[w_idx][2]
            else:
                is_target = '_'
                label = '_'
                frame = ''
            file.write(f"{word}\t{is_target}\t{label}\t{frame}\n")
        file.write('\n')
    file.close()

    # for index, batch in enumerate(tqdm(dl)):
    #     batch = {k: v.to(model.device) for k, v in batch.items()}
    #     word_index = batch.pop('word_index')
    #     word_ids = batch.pop('words_ids')
    #     mlabels = batch.pop('label')
    #     outputs = model(**batch)
    #     logits = outputs.logits
    #     labels = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

    #     for words, label, mlabel, word_idx, word_id in zip(batch['input_ids'], labels, mlabels, word_index, word_ids):
    #         words = tokenizer.convert_ids_to_tokens(words)
    #         label = [id2label[i.item()] for i in label]
    #         words = [word.strip('Ġ') for word in words]
    #         word_id = [str(i.item()) if i is not None else str(i) for i in word_id]
    #         file.write('token\t' + '\t'.join(words)+'\n')
    #         file.write('frame\t' + '\t'.join(label)+'\n')
    #         file.write('word_id\t' + '\t'.join(word_id)+'\n')
    #         file.write('target index\t' + str(word_idx.item())+'\n')
    #         file.write('metaphor label\t' + str(mlabel.item())+'\n')

    #     if index>200:
    #         break
    
    # file.close()
        # print(outputs)
        # break