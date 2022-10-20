# pip install -q datasets
from datasets import load_dataset
ds = load_dataset("bigscience/xP3", "code", use_auth_token="YOUR_AUTH_KEY")


def count_comments(x):
    x["counts"] = x["targets"].count("#")
    return x
counts = ds["train"].map(lambda x: count_comments(x))

char_count = sum([len(t) for t in counts["targets"]])
char_avg = char_count / len(counts)

comment_count = sum(counts["counts"])
comment_avg = comment_count / len(counts)

print(f"Char Avg {char_avg} ; Comment Avg {comment_avg}")
# Char Avg 530.5055696223101 ; Comment Avg 0.8540124473543094

