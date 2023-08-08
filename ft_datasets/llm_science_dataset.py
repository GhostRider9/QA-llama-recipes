import pandas as pd
from datasets import Dataset, DatasetDict
from .utils import Concatenator


def get_preprocessed_llm_science(dataset_config, tokenizer, split='train'):

    dataset = Dataset.from_pandas(pd.read_csv("data/new_train_df.csv").fillna(''))
    
    choice_prefixes = [chr(ord('A') + i) for i in range(26)] # A-Z

    prompt = '''Context: {hint}\nQuestion: {question}\nOptions: {options}\n---\nAnswer:{answer}{eos_token}'''

    def format_options(options):
        return '\n'.join([f'({c}) {o}' for c, o in zip(choice_prefixes, options)])

    def apply_prompt_template(r):
        options = format_options([r[x] for x in list('ABCDE')])
        return {
            "text": prompt.format(
                hint=r['wiki_fact'].replace('\n\n',' '),
                question=r['prompt'],
                options=options,
                answer=f"({r['answer']})",
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset