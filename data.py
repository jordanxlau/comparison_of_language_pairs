import random
import string
import re
import pandas as pd

from keras import layers

from tensorflow import data as tf_data
from tensorflow import strings as tf_strings

# Set a random seed, for reproducibility
random.seed(58)

lang1 = 'en' #arabic
lang2 = 'nl' #english

# Downloading the data
text = pd.read_parquet("hf://datasets/Helsinki-NLP/opus-100/"+lang1+"-"+lang2+"/train-00000-of-00001.parquet")

# Limit to 200 000 examples
text = text.head(500000)

# Parsing the data
text_pairs = []

for row in text["translation"]:
    l1 = lang1
    l2 = lang2
    #print("ROW[lang1]: " + "\'"+row[lang1]+"\'")
    l1 = row[lang1]
    l2 = row[lang2]

    l2 = "[start] " + l2 + " [end]"
    text_pairs.append((l1, l2))

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


lang1_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
lang2_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_lang1_texts = [pair[0] for pair in train_pairs]
train_lang2_texts = [pair[1] for pair in train_pairs]
lang1_vectorization.adapt(train_lang1_texts)
lang2_vectorization.adapt(train_lang2_texts)

# Formatting Datasets
def format_dataset(l1, l2):
    l1 = lang1_vectorization(l1)
    l2 = lang2_vectorization(l2)
    return (
        {
            "encoder_inputs": l1,
            "decoder_inputs": l2[:, :-1],
        },
        l2[:, 1:],
    )


def make_dataset(pairs):
    lang1_texts, lang2_texts = zip(*pairs)
    lang1_texts = list(lang1_texts)
    lang2_texts = list(lang2_texts)
    dataset = tf_data.Dataset.from_tensor_slices((lang1_texts, lang2_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")