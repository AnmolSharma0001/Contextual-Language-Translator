import numpy as np
import pandas as pd
import tensorflow as tf
import string
from string import digits
import re
import tensorflow_text as tf_text

df = pd.DataFrame(columns=['hindi', 'english'])


def clean_english_data(sentence):
    exclude = set(string.punctuation)
    remove_digits = str.maketrans('', '', string.digits)
    sentence = sentence.lower()
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = sentence.translate(remove_digits)
    sentence = sentence.strip()
    sentence = re.sub(" +", " ", sentence)
    return sentence


def clean_hindi_data(sentence):
    exclude = set(string.punctuation)
    remove_digits = str.maketrans('', '', string.digits)
    sentence = sentence.lower()
    sentence = ''.join(ch for ch in sentence if ch not in exclude)

    sent_temp = ''
    for c in sentence:
        if c == ' ':
            sent_temp += c
        elif ord(u'\u0900') <= ord(c) <= ord(u'\u097F'):
            sent_temp += c
    sentence = sent_temp

    sentence = re.sub('[a-z]', '', sentence)
    sentence = re.sub('[०१२३४५६७८९।]', '', sentence)
    sentence = sentence.translate(remove_digits)
    sentence = sentence.strip()
    sentence = re.sub(" +", " ", sentence)
    return sentence


def tf_lower_and_split_punct(text):
    # Split accented characters.

    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    text = tf_text.normalize_utf8(text, 'NFD')
    return text


lines = pd.read_csv("./final_dataset_1.csv", encoding='utf-8')

lines = lines[~pd.isnull(lines['english'])]
lines = lines[~pd.isnull(lines['hindi'])]

lines.drop_duplicates(inplace=True)

p = 0
q = 100000

ex = 0

for i in range(p+1, q):
    hindi = np.array(lines['hindi'][p:i])
    english = np.array(lines['english'][p:i])
    p += 1

    context_raw = np.array([clean_english_data(x) for x in english])
    target_raw = np.array([clean_hindi_data(x) for x in hindi])

    # Creating Tensorflow dataset
    # BUFFER_SIZE = len(context_raw)
    BATCH_SIZE = 1

    is_train = np.random.uniform(size=len(target_raw)) < 0.8

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((context_raw, target_raw))
        # .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE))
    
    max_vocab_size = 5000

    context_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size,
        ragged=True)

    context_text_processor.adapt(
        train_raw.map(lambda context, target: context))

    target_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size,
        ragged=True)

    target_text_processor.adapt(train_raw.map(lambda context, target: target))
    try:
        target_text_processor.get_vocabulary()
        print(">> Data clear at ", i)
        new_row = pd.Series({'hindi': hindi[0], 'english': english[0]})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    except:
        print(">> An exception occurred at ", i)
        ex += 1

print("Total Exceptions: ", ex)

df.to_csv('./hindi_to_english_dataset.csv',
          mode='a', index=False, header=False)
# df.to_csv('./hindi_to_english_dataset.csv', index = False)
