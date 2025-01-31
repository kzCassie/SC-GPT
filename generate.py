from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from common.utils import init_arg_parser, check_config, logger, set_seed


# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_t5=False, is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu',tokenizer=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context

    with torch.no_grad():
        if is_t5:
            output = model.generate(context)
            print(f"output shape={output.shape}")
            generated = torch.cat((generated, output), dim=1)
            print(f"generated shape={output.shape}")
            print(tokenizer.decode(output[0]))
        else:
            # TODO: use transformer lib
            for _ in range(length):
                inputs = {'input_ids': generated}

                if is_xlnet:
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

                if is_xlm_mlm and xlm_mask_token:
                    # XLM MLM models are direct models (predict same token, not next token)
                    # => need one additional dummy token in the input (will be masked and guessed)
                    input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                    inputs = {'input_ids': input_ids}

                if xlm_lang is not None:
                    inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for i in range(num_samples):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= repetition_penalty

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

    return generated


def decode(args, model, tokenizer):
    model.eval()

    # if args.length < 0 and model.config.max_position_embeddings > 0:
    #     args.length = model.config.max_position_embeddings
    # elif 0 < model.config.max_position_embeddings < args.length:
    #     args.length = model.config.max_position_embeddings  # No generation bigger than model size
    # elif args.length < 0:
    #     args.length = MAX_LENGTH  # avoid infinite loop

    # Cassie: modified for T5Config
    if hasattr(model.config, 'max_position_embeddings'):
        if args.length > 0:
            args.length = min(args.length, model.config.max_position_embeddings)
        else:
            args.length = min(MAX_LENGTH,  model.config.max_position_embeddings)  # avoid infinite loop
    else:
        args.length = min(args.length, MAX_LENGTH)

    ## log
    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    fin = open(args.decode_input_file)
    inputs = [i.strip() for i in fin]

    output_tests = []  # List(List(Str)): list of top generated examples
    for idx in range(0, len(inputs), 1):
        logger.info(f"PROGRESS: {int(idx/len(inputs)*100)}%")
        xlm_lang = None
        # XLM Language usage detailed in the issues #1414
        if args.model_type in ["xlm"] and hasattr(tokenizer, 'lang2id') and hasattr(model.config, 'use_lang_emb') \
                and model.config.use_lang_emb:
            if args.xlm_lang:
                language = args.xlm_lang
            else:
                language = None
                while language not in tokenizer.lang2id.keys():
                    language = input("Using XLM. Select language in " + str(list(tokenizer.lang2id.keys())) + " >>> ")
            xlm_lang = tokenizer.lang2id[language]

        # XLM masked-language modeling (MLM) models need masked token (see details in sample_sequence)
        is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
        if is_xlm_mlm:
            xlm_mask_token = tokenizer.mask_token_id
        else:
            xlm_mask_token = None

        # raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        lines = inputs[idx]
        raw_text = lines.split(' & ')[0] + ' & '
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text

        print(raw_text)
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)  # raw_text token IDs

        if args.model_type == "ctrl":
            if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
        out = sample_sequence(
            model=model,
            context=context_tokens,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            is_t5=bool(args.model_type == 't5'),
            is_xlnet=bool(args.model_type == "xlnet"),
            is_xlm_mlm=is_xlm_mlm,
            xlm_mask_token=xlm_mask_token,
            xlm_lang=xlm_lang,
            device=args.device,
            tokenizer=tokenizer
        )
        out = out[:, len(context_tokens):].tolist()
        examples = []
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            examples.append(text)
            print(text)
        
        output_tests.append(examples)
        break  # TODO
        # if args.prompt:
            # break

    import json
    json.dump(output_tests, open(args.decode_output_file, 'w'), indent=2)
    return text


# if __name__ == '__main__':
    # command_line = """
    #                 --no_cuda
    #                 --mode decode
    #                 --model_type=t5 \
    #                 --model_path=saved_models/t5 \
    #                 --num_samples 5 \
    #                 --input_file=data/restaurant/test.txt \
    #                 --top_k 5 \
    #                 --output_file=saved_models/t5/results.json \
    #                 --length 80
    #                 """
    # parser = init_arg_parser()
    # args = parser.parse_args(command_line.split())

