"""Parse command line arguments and validate their eligibility."""
import argparse
import os
import re
import shutil
import glob
import random
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def init_arg_parser():
    parser = argparse.ArgumentParser()

    ### General configuration ###
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument("--mode", default="", type=str, choices=["train", "eval", "train_with_eval", "decode"],
                        required=True, help="Choose the running mode for the script.")
    parser.add_argument("--evaluate_during_training", default=False, action="store_true",
                        help="Evaluating during training")

    parser.add_argument("--model_type", choices=['gpt2', 'openai-gpt', 'bert', 'roberta', 'distilbert', 't5'],
                        default="bert", type=str, required=True, help="The model architecture to be initialized.")
    parser.add_argument("--model_name", default="", type=str,
                        help="The shortcut name of the pre-trained model weights for initialization.")
    parser.add_argument("--model_path", default="", type=str,
                        help="The path of model weights checkpoint for initialization.")

    ### Cuda & Distributed Training ###
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Do not use gpu")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    ### Training ###
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Training schedule details
    parser.add_argument("--block_size", default=80, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "When block_size<=0, use the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # TODO: after modification, must keep all things together in model_path
    # Load pre-train filepath
    # parser.add_argument("--config_name", default="", type=str,
    #                     help="Optional pretrained config name or path if not the same as model_name_or_path")
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 "
                             "(instead of the default one)")

    ### Evaluating ###
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    ### Decoding ###
    # TODO: write help strings
    parser.add_argument('--decode_input_file', type=str, default=None, help="file")
    parser.add_argument('--decode_output_file', type=str, default=None, help="file")

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")

    parser.add_argument('--stop_token', type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument('--nc', type=int, default=1, help="number of sentence")
    parser.add_argument("--use_token", action='store_true', help="")

    ### Model configuration ###
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mlm", default=False, action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    ### Data Processing ###
    # TODO: these options all related to data loading?
    parser.add_argument('--text_chunk', default=False, action='store_true',
                        help="Read the text data file in one chunk instead of reading it line by line")
    parser.add_argument('--use_reverse', default=False, action='store_true', help="")
    parser.add_argument('--with_code_loss', type=bool, default=True, help="")
    parser.add_argument('--use_tokenizer', default=False, action='store_true',
                        help="Use pretrained tokenizer. If false, do a simple split by space")
    parser.add_argument("--max_seq", default=80, type=int,
                        help="Max num tokens when loading text (including tokens for both code and utterance)")

    ### Logging and save ###
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, '
                             'does not delete by default')
    parser.add_argument("--eval_all_checkpoints", default=False, action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending "
                             "and ending with step number")
    parser.add_argument('--overwrite_output_dir', default=False, action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', default=False, action='store_true',
                        help="Overwrite the cached processed dataset for training and evaluation")

    ### Precision ###
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    ### Distant debugging ###
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    return parser


def check_config(parser):
    """ Perform sanity checks on command line parsed arguments"""
    command_line = """--output_dir=saved_models/t5
        --no_cuda
        --mode train
        --model_type=t5
        --model_name=t5-base
        --train_data_file=data/restaurant/train.txt 
        --eval_data_file=data/restaurant/train.txt 
        --num_train_epochs 5
        --learning_rate 5e-5 
        --use_tokenizer 
        --overwrite_output_dir
        --overwrite_cache
        """

    command_line2 = """
                --no_cuda
                --mode decode
                --model_type=t5 \
                --model_path=saved_models/t5 \
                --num_samples 5 \
                --decode_input_file=data/restaurant/test.txt \
                --top_k 5 \
                --decode_output_file=saved_models/t5/results.json \
                --length 80
                """
    args = parser.parse_args()
    # args = parser.parse_args(command_line.split())
    # args = parser.parse_args(command_line2.split())

    ### generic config ###
    if args.model_type in ['t5']:
        args.enc_dec = True

    # which pre-trained model to load
    if not args.model_name and not args.model_path:
        raise ValueError("Must specify either --model_name or --model_path")
    elif args.model_name and args.model_path:
        raise ValueError("Only specify either model_name or model_path")
    else:
        args.model_loc = args.model_name if args.model_name else args.model_path

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    ## train
    if 'train' in args.mode:
        if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
            raise ValueError("BERT, RoBERTa and distilbert do not have LM heads but masked LM heads. They must be run using the --mlm "
                             "flag (masked language modeling).")



        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

        # Setup CUDA, GPU & distributed training
        # local_rank=-1: disable distributed machine with node structure
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.n_gpu = 1
        args.device = device

        # Set seed
        set_seed(args)

        # Setup distant debugging if needed
        if args.server_ip and args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        # logger
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    ## decode
    if args.mode == "decode":
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        set_seed(args)

    return args


def rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def load_pretrained(args, config_class, model_class, tokenizer_class):
    ## Load pretrained model and tokenizer ##
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config = config_class.from_pretrained(args.model_loc, cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_loc, from_tf=bool('.ckpt' in args.model_loc), config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.model_loc, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)

    args.block_size = tokenizer.max_len_single_sentence if args.block_size <= 0 \
        else min(args.block_size, tokenizer.max_len_single_sentence)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    return config, model, tokenizer


def save_pretrained(args, config, model, tokenizer):
    # Saving best-practices:
    # if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        config.save_pretrained(args.output_dir)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


