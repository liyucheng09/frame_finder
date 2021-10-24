from transformers import RobertaForTokenClassification, BertForTokenClassification
import os
from lyc.utils import dataset_wrapper

class FrameFinder(RobertaForTokenClassification):
    def forward(self, parse_tree=None, **kwargs):
        output = super().forward(**kwargs)

        if parse_tree is None:
            return output

        logits = output.logits
        embeddings = output.hidden_states[-1]

        aspect_index = parse_tree.aspect_index
        aspect_embedding = embeddings[aspect_index]

        aspect_children_index = parse_tree.children_dirs
        children_embedding = embeddings[aspect_children_index]

        return aspect_embedding, children_embedding

