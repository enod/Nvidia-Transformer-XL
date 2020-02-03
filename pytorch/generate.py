import os
import torch

def load_checkpoint(path):
  if os.path.isdir(path):
    path = os.path.join(path, 'checkpoint_best.pt')
  dst = f'cuda:{torch.cuda.current_device()}'
  print(f'Loading checkpoint from {path}')
  checkpoint = torch.load(path, map_location=dst)
  return checkpoint


ckpt = load_checkpoint("./LM-TFM/")

# # adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
cutoffs = [19997, 39997, 199997]
tie_projs += [True] * len(cutoffs)

model_config_base = {
  'dropout'       : 0.1,
  'dropatt'       : 0.0,
  'tie_weight'    : False,
  'div_val'       : 1,
  'pre_lnorm'     : True,
  'cutoffs'       : cutoffs,
  'clamp_len'     : 400,
  }

from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLConfig

# Initializing a Transformer XL configuration
configuration = TransfoXLConfig.from_dict(model_config_base)
# To match with pre-trained model
configuration.d_embed, configuration.d_head  = 512, 64
configuration.d_inner, configuration.d_model = 2048, 512
configuration.mem_len, configuration.n_head = 192, 8
configuration.n_layer, configuration.tgt_len = 16, 192
configuration.vocab_size = 32000

model = TransfoXLLMHeadModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=ckpt['model_state'], config=configuration)

from transformers import PreTrainedTokenizer
from utils.tokenization_sentencepiece import FullTokenizer
from collections import Counter, OrderedDict
from os.path import join, exists


class Vocab(TransfoXLTokenizer):
    def __init__(        
        self,
        special=None,
        min_freq=0,
        max_size=None,
        lower_case=False,
        delimiter=None,
        vocab_file='./data/mn_cased.vocab',
        never_split=None,
        unk_token="<unk>",
        eos_token="</s>",
        additional_special_tokens=["<formula>"],
        **kwargs
        ):

        super().__init__(
            unk_token=unk_token, eos_token=eos_token, additional_special_tokens=additional_special_tokens, **kwargs
        )

        self.vocab_file = vocab_file
        if vocab_file is not None:
            self.build_vocab()

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        tokenizer = FullTokenizer(model_file=join('./data', 'mn_cased.model'),
                          vocab_file=join('./data', 'mn_cased.vocab'), do_lower_case=False)
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = tokenizer.tokenize(line)

        if add_double_eos:  # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<unk>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)
            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))
    
model.to('cuda')
model.eval()
model.half()
cool_tokenizer = Vocab()

# reference - https://github.com/huggingface/transformers/blob/2ba147ecffa28e5a4f96eebd09dcd642117dedae/examples/run_generation.py
def text_generation(prompt_text, temp, topk, topp, beams, penalty, do_sample):
    encoded_prompt = cool_tokenizer.encode(prompt_text, return_tensors="pt")
    encoded_prompt = encoded_prompt.to('cuda')

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=20,
        temperature=temp,
        top_k=topk,
        top_p=topp,
        num_beams=beams,
        repetition_penalty=penalty,
        do_sample=do_sample,
    )

    # Batch size == 1. to add more examples please use num_return_sequences > 1
    generated_sequence = output_sequences[0].tolist()
    text = cool_tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text = [word.replace("▁", " ") for word in text.split()]

    return ' '.join(text)

print(text_generation("УИХ ", temp=1.0, topk=5, topp=1, beams=1, penalty=1.0, do_sample=True))