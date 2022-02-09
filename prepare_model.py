import argparse
import os
import json

import sentencepiece as spm

from transformers import (
    AutoTokenizer,
    MarianTokenizer,
    EncoderDecoderModel
)

def main():
    parser = argparse.ArgumentParser(description='My App.')
    parser.add_argument('--vocab_file', type=str, required=True, help='path of text file containing the target vocabulary')
    parser.add_argument('--source_model', type=str, required=True, help='name or path of the source model')
    parser.add_argument('--target_model', type=str, required=True, help='name or path of the target model')
    parser.add_argument('--final_name', type=str, required=True, help='name of the final model')
    parser.add_argument('--temp_spm_name', type=str, default='tmp_spm')
    parser.add_argument('--temp_vocab_name', type=str, default='tmp_voc')
    parser.add_argument('--outputs_dir', type=str, default='./')
    parser.add_argument('--model_max_length', type=int, default=512)
    args = parser.parse_args()
    
    ##
    # prepare source tokenizer
    ##

    os.makedirs(args.outputs_dir, exist_ok=True)

    src_tokenizer = AutoTokenizer.from_pretrained(args.source_model, use_fast=False)
    
    src_voc_path = os.path.join(args.outputs_dir, args.temp_vocab_name) + '_trg.json'
    with open(src_voc_path, 'w+') as f:
        json.dump(src_tokenizer.get_vocab(), f)   
    
    src_spm_path, = src_tokenizer.save_vocabulary('./')
    
    ##
    # prepare target tokenizer
    ##
    
    trg_tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=False)
    
    with open(args.vocab_file, 'r') as f:
        vocab = f.read().splitlines()
    
    trg_spm_path = os.path.join(args.outputs_dir, args.temp_spm_name) + '_trg'
    
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(set(vocab)),
        model_prefix=trg_spm_path, 
        model_type='word', 
        character_coverage=1.0, 
        split_by_whitespace=False,
        split_by_unicode_script=False,
        split_by_number=False,
        hard_vocab_limit=False,
        unk_id=src_tokenizer.unk_token_id,
        pad_id=src_tokenizer.pad_token_id,
        eos_id=src_tokenizer.eos_token_id if src_tokenizer.eos_token is not None else src_tokenizer.sep_token_id,
        unk_piece=src_tokenizer.unk_token,
        pad_piece=src_tokenizer.pad_token,
        eos_piece=src_tokenizer.eos_token if src_tokenizer.eos_token is not None else src_tokenizer.sep_token,
    )
    
    ##
    # combine source and target tokenizers
    ##

    tokenizer = MarianTokenizer(
            vocab=src_voc_path,
            source_spm=src_spm_path,
            target_spm=trg_spm_path + '.model',
            source_lang='multi',
            target_lang='sign',
            unk_token=src_tokenizer.unk_token,
            eos_token=src_tokenizer.eos_token,
            pad_token=src_tokenizer.pad_token,
            model_max_length=args.model_max_length,
    )
    
    model_dir = os.path.join(args.outputs_dir, args.final_name)
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(model_dir)

    for file in [src_spm_path, src_voc_path, trg_spm_path + '.model', trg_spm_path + '.vocab']:
        os.remove(file)
        
    ##
    # prepare combined model
    ##
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
         args.source_model, args.target_model,
    )  
    
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    model.config.tokenizer_class = tokenizer.__class__.__name__
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = args.model_max_length
    model.config.decoder_start_token_id = trg_tokenizer.bos_token_id
    
    model.save_pretrained(model_dir)

if __name__ == '__main__':
    main()