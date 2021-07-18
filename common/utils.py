"""Parse command line arguments and validate their eligibility."""
import argparse

# def init_train_arg_parser():
#     parser = argparse.ArgumentParser()
#
#     ## Training ##
#     parser.add_argument("--train_data_file", default=None, type=str, required=True,
#                         help="The input training data file (a text file).")
#     parser.add_argument("--output_dir", default=None, type=str, required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")
#
#     ## General config ##
#     parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
#     parser.add_argument("--no_cuda", default=False, action='store_true', help="Use gpu")
#     parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
#
#     parser.add_argument("--model_type", choices=['gpt2', 'openai-gpt', 'bert', 'roberta', 'distilbert', 't5'],
#                         default="bert", type=str,
#                         help="The model architecture to be fine-tuned.")
#     # parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
#     #                     help="The model checkpoint for weights initialization.")

#     parser.add_argument("--model_name", default="bert-base-cased", type=str,
#                         help="The name of the pre-trained model for weights initialization.")
#     parser.add_argument("--model_path", default="", type=str,
#                         help="The model checkpoint for weights initialization.")
#
#     parser.add_argument("--mlm", default=False, action='store_true',
#                         help="Train with masked-language modeling loss instead of language modeling.")
#     parser.add_argument("--mlm_probability", type=float, default=0.15,
#                         help="Ratio of tokens to mask for masked language modeling loss")
#
#     ## Mode ##
#     parser.add_argument("--do_train", default=False, action='store_true',
#                         help="Whether to run training.")
#     parser.add_argument("--do_eval", default=False, action='store_true',
#                         help="Whether to run eval on the dev set.")
#     parser.add_argument("--evaluate_during_training", default=False, action='store_true',
#                         help="Run evaluation during training at each logging step.")
#     parser.add_argument("--do_lower_case", default=False, action='store_true',
#                         help="Set this flag if you are using an uncased model.")
#
#     ## Load pre-train filepath ##
#     parser.add_argument("--config_name", default="", type=str,
#                         help="Optional pretrained config name or path if not the same as model_name_or_path")
#     parser.add_argument("--tokenizer_name", default="", type=str,
#                         help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
#     parser.add_argument("--cache_dir", default="", type=str,
#                         help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
#
#     ## Evaluate ##
#     parser.add_argument("--eval_data_file", default=None, type=str,
#                         help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
#
#     ## Training hyper-param ##
#     parser.add_argument("--block_size", default=80, type=int,
#                         help="Optional input sequence length after tokenization."
#                              "The training dataset will be truncated in block of this size for training."
#                              "When block_size<=0, use the model max input length for single sentence inputs "
#                              "(take into account special tokens).")
#     parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
#                         help="Batch size per GPU/CPU for training.")
#     parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
#                         help="Batch size per GPU/CPU for evaluation.")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--learning_rate", default=5e-5, type=float,
#                         help="The initial learning rate for Adam.")
#     parser.add_argument("--weight_decay", default=0.0, type=float,
#                         help="Weight decay if we apply some.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")
#     parser.add_argument("--num_train_epochs", default=1.0, type=float,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--max_steps", default=-1, type=int,
#                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
#     parser.add_argument("--warmup_steps", default=0, type=int,
#                         help="Linear warmup over warmup_steps.")
#     parser.add_argument('--text_chunk', default=False, action='store_true', help="")    # TODO: add help string
#     parser.add_argument('--use_reverse', default=False, action='store_true', help="")
#     parser.add_argument('--with_code_loss', type=bool, default=True, help="")
#     parser.add_argument('--use_tokenize', default=False, action='store_true', help="")
#     parser.add_argument("--max_seq", default=80, type=int, help="")  # TODO: --max_seq vs --block_size?
#
#     ## logging and save ##
#     parser.add_argument('--logging_steps', type=int, default=100,
#                         help="Log every X updates steps.")
#     parser.add_argument('--save_steps', type=int, default=5000,
#                         help="Save checkpoint every X updates steps.")
#     parser.add_argument('--save_total_limit', type=int, default=None,
#                         help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
#     parser.add_argument("--eval_all_checkpoints", default=False, action='store_true',
#                         help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
#     parser.add_argument('--overwrite_output_dir', default=False, action='store_true',
#                         help="Overwrite the content of the output directory")
#     parser.add_argument('--overwrite_cache', default=False, action='store_true',
#                         help="Overwrite the cached training and evaluation sets")
#
#     ## Precision ##
#     parser.add_argument('--fp16', default=False, action='store_true',
#                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
#     parser.add_argument('--fp16_opt_level', type=str, default='O1',
#                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#                              "See details at https://nvidia.github.io/apex/amp.html")
#
#     ## Distant debugging ##
#     parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
#     parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
#
#     return parser


# def init_gen_arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_type", default=None, type=str, required=True,
#                         help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
#     parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
#                         help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
#     parser.add_argument("--prompt", type=str, default="")
#     parser.add_argument("--padding_text", type=str, default="")
#     parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
#     parser.add_argument("--length", type=int, default=40)
#     parser.add_argument("--num_samples", type=int, default=1)
#     parser.add_argument("--temperature", type=float, default=1.0,
#                         help="temperature of 0 implies greedy sampling")
#     parser.add_argument("--repetition_penalty", type=float, default=1.0,
#                         help="primarily useful for CTRL model; in that case, use 1.2")
#     parser.add_argument("--top_k", type=int, default=0)
#     parser.add_argument("--top_p", type=float, default=0.9)
#     parser.add_argument("--no_cuda", action='store_true',
#                         help="Avoid using CUDA when available")
#     parser.add_argument('--seed', type=int, default=42,
#                         help="random seed for initialization")
#     parser.add_argument('--stop_token', type=str, default=None,
#                         help="Token at which text generation is stopped")
#
#     parser.add_argument('--input_file', type=str, default=None,
#                         help="file")
#     parser.add_argument('--output_file', type=str, default=None,
#                         help="file")
#
#     parser.add_argument('--nc', type=int, default=1,
#                         help="number of sentence")
#
#     parser.add_argument("--use_token", action='store_true',
#                         help="Avoid using CUDA when available")
#
#     # parser.add_argument('--use_token', type=int, default=1,
#     # help="number of sentence")
#
#     return parser


def init_arg_parser():
    parser = argparse.ArgumentParser()

    ### General configuration ###
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument("--mode", default="", type=str, choices=["train", "eval"], required=True,
                        help="Choose the running mode for the script.")

    parser.add_argument("--model_type", choices=['gpt2', 'openai-gpt', 'bert', 'roberta', 'distilbert', 't5'],
                        default="bert", type=str, required=True, help="The model architecture to be initialized.")
    parser.add_argument("--model_name", default="bert-base-cased", type=str,
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
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--train_output_dir", default=None, type=str, required=True,
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
    # parser.add_argument("--cache_dir", default="", type=str,
    #                     help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")

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

    parser.add_argument('--text_chunk', default=False, action='store_true', help="")  # TODO: add help string
    parser.add_argument('--use_reverse', default=False, action='store_true', help="")
    parser.add_argument('--with_code_loss', type=bool, default=True, help="")
    parser.add_argument('--use_tokenize', default=False, action='store_true', help="")
    parser.add_argument("--max_seq", default=80, type=int, help="")  # TODO: --max_seq vs --block_size?

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
                        help="Overwrite the cached training and evaluation sets")

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


def val_train_config(parser):
    """ perform sanity checks on command line parsed arguments for training"""
    command_line = """--output_dir=saved_models/t5
        --no_cuda
        --mode train 
        --model_type=t5
        --model_name=t5-base
        --train_data_file=data/restaurant/train.txt 
        --eval_data_file=data/restaurant/train.txt 
        --per_gpu_train_batch_size 1 
        --num_train_epochs 20
        --learning_rate 5e-5 
        --use_tokenize 
        --overwrite_output_dir
        --overwrite_cache
        """
    args = parser.parse_args(command_line.split())

    if args.do_train:
        if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
            raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                             "flag (masked language modeling).")

        if args.eval_data_file is None and args.do_eval:
            raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                             "or remove the --do_eval argument.")

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
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

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    return args

