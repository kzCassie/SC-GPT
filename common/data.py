"""Data loader for FewShotWoz dataset."""
import os
import torch
import pickle

from torch.utils.data import Dataset
from common.utils import logger


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, max_seq=80):
        assert os.path.isfile(file_path)
        directory, tail = os.path.split(file_path)
        filename, ext = os.path.splitext(tail)
        cached_features_file = os.path.join(directory, args.model_type + '_cached_lm_' + str(
            block_size) + '_seqlen_' + str(max_seq) + '_' + filename + '.bin')

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []

            with open(file_path, encoding="utf-8") as f:
                if args.text_chunk:
                    text = f.read()
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                else:
                    for line in f:
                        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line.strip() + ' ' + tokenizer.eos_token))
                        self.examples.append(tokenized_text)

            if args.text_chunk:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


class TextSeqDataset(Dataset):
    """
    Input Text Format: <Dialogue Act> + & + <Natural Language Utterance>.
                        One instance per line.
    Output Text Format:
        enc_dec=False: List of tokens for the entire line, i.e. the tokens include the DA code, the
                        separator '&' and the tokenized utterance.
                        Dataloader for 'Language Modeling' type of model such as GPT.
        enc_dec=True: Dataloader for encoding-decoding type of model.
    """
    def __init__(self, tokenizer, args, file_path, max_seq=80, separator=' & ', enc_dec=False):
        assert os.path.isfile(file_path)
        directory, tail = os.path.split(file_path)
        filename, ext = os.path.splitext(tail)
        cached_features_file = os.path.join(directory, args.model_type + '_encdec_' + str(enc_dec).lower() + '_seqlen_' + str(max_seq) + '_' +
                                            filename + '.bin')

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples, self.masks, self.labels = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file %s", file_path)
            self.examples = []
            self.labels = []
            self.masks = []
            with open(file_path, encoding="utf-8") as f:
                """ Normalize all lines to max_len number of tokens. Directly truncate (i.e. discard) tokens that 
                    exceed the max_seq limit. 
                """
                for line in f:
                    # truncate raw_str to max_seq tokens
                    # append eos token at the end
                    raw_str = line.strip().lower()
                    raw_str_token = raw_str.split()  #TODO: text data is always lower cased. modify default cmd line argument?
                    if len(raw_str_token) > max_seq - 1:
                        raw_str = ' '.join(raw_str_token[:max_seq - 1])
                    raw_str += ' ' + tokenizer.eos_token

                    # Tokenize
                    if enc_dec:  # dataloader for encoding/decoding model
                        str_split = raw_str.split(separator)
                        code_str = str_split[0]
                        utter_str = str_split[1]

                        if args.use_tokenizer:
                            tokenized_code = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code_str))
                            tokenized_utter = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utter_str))
                        else:
                            tokenized_code = tokenizer.convert_tokens_to_ids(code_str.split())
                            tokenized_utter = tokenizer.convert_tokens_to_ids(utter_str.split())

                        self.examples.append(tokenized_code)  # TODO: pad text?
                        self.labels.append(tokenized_utter)
                        self.masks.append([])
                    else:  # dataloader for Language Modeling model
                        if args.use_tokenizer:
                            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                        else:
                            tokenized_text = tokenizer.convert_tokens_to_ids(raw_str.split())

                        label = [-1] * max_seq
                        label[:len(tokenized_text)] = tokenized_text
                        mask = [1] * max_seq

                        if len(tokenized_text) < max_seq:
                            mask[-(max_seq - len(tokenized_text)):] = [0] * (max_seq - len(tokenized_text))
                            # label[code_str_len:len(tokenized_text)] = tokenized_text[code_str_len:]
                            tokenized_text = tokenized_text + [0] * (max_seq - len(tokenized_text))
                        else:
                            tokenized_text = tokenized_text[:max_seq]
                            # label[code_str_len:] = tokenized_text[code_str_len:]

                        self.examples.append(tokenized_text)
                        self.masks.append(mask)
                        self.labels.append(label)

            # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            if args.with_code_loss:  # default: True
                self.labels = self.examples
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump((self.examples, self.masks, self.labels), handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.masks[item]), torch.tensor(self.labels[item])


def load_and_cache_examples(args, tokenizer, evaluate=False, enc_dec=False):
    dataset = TextSeqDataset(tokenizer, args, max_seq=args.max_seq, enc_dec=enc_dec,
                             file_path=args.eval_data_file if evaluate else args.train_data_file)
    return dataset
