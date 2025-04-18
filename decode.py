import random

from data import lang1_vectorization, lang2_vectorization, test_pairs, lang1, lang2
from keras import ops, models
import keras_nlp

from utils import PositionalEmbedding, TransformerEncoder, TransformerDecoder

# BLEU Metric
bleu_metric = keras_nlp.metrics.Bleu()
# Get saved model
transformer = models.load_model(lang1+"-"+lang2+"-v1.keras", custom_objects={
    'PositionalEmbedding': PositionalEmbedding,
    'TransformerEncoder': TransformerEncoder,
    'TransformerDecoder': TransformerDecoder
})

# Decoding test sentences
lang2_vocab = lang2_vectorization.get_vocabulary()
lang2_index_lookup = dict(zip(range(len(lang2_vocab)), lang2_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = lang1_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = lang2_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer(
            {
                "encoder_inputs": tokenized_input_sentence,
                "decoder_inputs": tokenized_target_sentence,
            }
        )

        # ops.argmax(predictions[0, i, :]) is not a concrete value for jax here
        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        sampled_token = lang2_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence

# Evaluate on test set
ground_truths = []
predictions = []

test_inputs = [pair[0] for pair in test_pairs]
test_references = [pair[1] for pair in test_pairs]
print(len(test_inputs))

for i in range(len(test_inputs)):
    input_sentence = test_inputs[i]
    reference_sentence = test_references[i]
    translated_sentence = decode_sequence(input_sentence)
    predictions.append(translated_sentence.split())  # Tokenize prediction
    ground_truths.append([reference_sentence.split()])  # Tokenized reference
    print(f"{input_sentence}: {reference_sentence} --> {translated_sentence}\n")
    print(str(i) + "/75000")
    
print("done populating ground truths and predictions")

# Compute BLEU score
bleu_score = bleu_metric._calculate_bleu_score(ground_truths, predictions)
print(f"BLEU Score: {bleu_score[0]:.4f}")

