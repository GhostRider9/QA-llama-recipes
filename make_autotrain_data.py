import pandas as pd

def format_options(options):
    return '\n'.join([f'({c}) {o}' for c, o in zip(list('ABCDE'), options)])

df = pd.read_csv("data/new_train_df.csv").fillna('')

ptompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

### Instruction:
According to the context, select the most accurate answer for the quesiton.
### Question:
{prompt}
### Options:
{options}
### Context:
{context}
[/INST]
The most accurate answer is ("""

output_template = '{answer}'

data = []
for _, r in df.iterrows():
    options = format_options([r[x] for x in list('ABCDE')])
    input = ptompt_template.format(prompt=r['prompt'], context=r['wiki_fact'].replace('\n',''),options=options)
    # input = ptompt_template.format(prompt=r['prompt'],options=options)
    otuput = output_template.format(answer=r['answer'])
    text = input + otuput
    data.append((input, otuput, text))

output_df = pd.DataFrame(data, columns=['input', 'output', 'text'])
test_df, train_df = output_df.iloc[:200], output_df.iloc[200:]
train_df.to_csv('./data/autotrain/train.csv', index=False)
test_df.to_csv('./data/autotrain/valid.csv', index=False)