import datasets
from .utils import Concatenator


def get_preprocessed_sciqa(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("metaeval/ScienceQA_text_only", split=split)
    choice_prefixes = [chr(ord('A') + i) for i in range(26)] # A-Z

    prompt = '''Context: {hint}\nQuestion: {question}\nOptions: {options}\n---\nAnswer:{answer}{eos_token}'''

    def format_options(options):
        return '\n'.join([f'({c}) {o}' for c, o in zip(choice_prefixes, options)])

    def apply_prompt_template(r):
        options = format_options(r['choices'])
        return {
            "text": prompt.format(
                hint=r["hint"],
                question=r["question"],
                options=options,
                answer=choice_prefixes[r['answer']],
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
