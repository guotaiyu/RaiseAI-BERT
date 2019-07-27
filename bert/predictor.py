from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from inputprocessor import InputExample, PaddingInputExample, InputFeatures
import tokenization
import tensorflow as tf
import requests

max_seq_length = 128
vocab_file = 'vocab.txt'
tf_serving_url = 'http://localhost:9000/v1/models/bert:predict'
label_list = ["contradiction", "entailment", "neutral"]

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(example, label_list, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  tf.compat.v1.logging.info("*** Input ***")
  tf.compat.v1.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=0,
      is_real_example=True)
  return feature


def predict(text1, text2):
  
  #1 create input example from two given texts
  input_example = InputExample(text_a = text1, text_b = text2)
    
  #2 create input features from inut example
  input_features = convert_single_example(input_example, label_list, max_seq_length, tokenizer)

  #3 create input dict from input features
  input_dict = {"input_ids": input_features.input_ids,"input_mask": input_features.input_mask,"segment_ids": input_features.segment_ids,"label_ids": input_features.label_id}
    
  #4 receive response from tensorflow serving with fine-tuned bert model
  response = requests.post(tf_serving_url, json={"inputs": input_dict}).json()

  tf.compat.v1.logging.info("*** Output ***") 
  tf.compat.v1.logging.info(response)
  
  return dict(zip(label_list, response["outputs"][0]))
