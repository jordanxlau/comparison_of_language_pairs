import random
import string
import re
import pandas as pd

from keras import layers

from tensorflow import data as tf_data
from tensorflow import strings as tf_strings

# Downloading the data
text = pd.read_parquet("hf://datasets/Helsinki-NLP/opus-100/af-en/train-00000-of-00001.parquet")

# Parsing the data
text_pairs = []

for row in text["translation"]:
    en = row['en']
    af = row['af']

    af = "[start] " + af + " [end]"
    text_pairs.append((en, af))

for _ in range(5):
    print(random.choice(text_pairs))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

# Vectorizing the text data
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64


def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


eng_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
af_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_af_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
af_vectorization.adapt(train_af_texts)

# Formatting Datasets
def format_dataset(eng, af):
    eng = eng_vectorization(eng)
    af = af_vectorization(af)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": af[:, :-1],
        },
        af[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, af_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    af_texts = list(af_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, af_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")