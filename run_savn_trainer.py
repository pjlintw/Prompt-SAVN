import logging
import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from transformers import (WEIGHTS_NAME, 
    BertForMaskedLM,
    RobertaConfig,
    RobertaTokenizer,
    BertConfig, 
    BertTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup)
from transformers.file_utils import is_apex_available

from sa_utils import (read_multiwoz_examples, convert_examples_to_features, get_slot_input_ids, multiwoz_evaluate,
                      RawResult, get_slots)
from sa_config import set_config

from my_utils import set_seed, tensor_to_list
from tqdm import tqdm, trange
import timeit
from models.slot_attention import SlotAttentionNetwork
from trainer_slot_attention import SlotAttentionNetworkTrainer

# 
PATTERN_TEMPLATE = {
    0: "###",
    1: "### . [MASK]",
    1: "### ? [MASK]",
    2: "what is the prediction for ### ? [MASK]",
}

# Mapping type_vale into token id by tokenizer(SLOT_TYPE_MAAPING["none"])
SLOT_TPYE_MAPPING = {
    "none": "none",
    "dontcare": "no",
    "span": "span"
}


logger = logging.getLogger(__name__)

# install apex from https://www.github.com/nvidia/apex to use fp16 training
# if multi-GPUs training
if is_apex_available():
    from apex import amp


def load_dataset(args, tokenizer, input_file, data_split_name=None, is_training=None, output_examples=False):
    """Loading dataset and convert to features.

    Steps:
        1. Constrcut cached examples and features files
        2. Load cached file if exists. Otherwise, create examples and features 

    Args:

    """
    assert data_split_name in {"train", "eval", "test"}
    
    ### Prepare cached file ###
    if args.use_pattern:
        cached_features_file_tmp = 'cached_pattern_features_{}_{}'
        cached_examples_file_tmp = 'cached_pattern_examples_{}_{}'
    else:
        cached_features_file_tmp = 'cached_features_{}_{}'
        cached_examples_file_tmp = 'cached_examples_{}_{}'
    
    cached_features_file = os.path.join(os.path.dirname(input_file), cached_features_file_tmp.format(
        data_split_name, args.utils_version))
    cached_examples_file = os.path.join(os.path.dirname(input_file), cached_examples_file_tmp.format(
        data_split_name, args.utils_version))

    slots = get_slots("2.1")
    logger.info(f"All slot {slots}")

    # Add suffix for training file
    if args.max_train_samples > 1 and args.max_train_samples is not None and is_training is True:
        cached_features_file += f"_{args.max_train_samples}"
        cached_examples_file += f"_{args.max_train_samples}"
    ### Prepare cached file ###

    ### File with small size of exmaples for debugging ###
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading %s features from cached file %s", data_split_name, cached_features_file)
        ### time it ###
        start_time = timeit.default_timer()
        examples = torch.load(cached_examples_file)
        exampleLoadingTime = timeit.default_timer() - start_time
        logger.info("Loading %s examples done in total %f secs (%f sec per example)", data_split_name, exampleLoadingTime, exampleLoadingTime / len(examples))
        
        start_time = timeit.default_timer()
        features = torch.load(cached_features_file)
        featureLoadingTime = timeit.default_timer() - start_time 
        logger.info("Loading %s features done in total %f secs (%f sec per example)", data_split_name, featureLoadingTime, featureLoadingTime / len(features))
        
    else:
        logger.info("Creating %s features from dataset file at %s", data_split_name, input_file)
        ### time it ###
        start_time = timeit.default_timer()
        examples = read_multiwoz_examples(input_file,
                                          is_training, 
                                          args.dataset_version,
                                          args.max_dialogue_size, 
                                          args.max_train_samples) # Only limit training exmaple
        exampleCreatingTime = timeit.default_timer() - start_time
        logger.info("Creating %s examples done in total %f secs (%f sec per example)", data_split_name, exampleCreatingTime, exampleCreatingTime / len(examples))

        start_time = timeit.default_timer()
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                is_training=is_training,
                                                use_sp=args.use_sp,
                                                use_pattern=args.use_pattern) 
        featureCreatingTime = timeit.default_timer() - start_time
        logger.info("Creating %s features done in total %f secs (%f sec per example)", data_split_name, featureCreatingTime, featureCreatingTime / len(features))
        
        logger.info("Saving %s examples into cached file %s", data_split_name, cached_examples_file)
        logger.info("Saving %s features into cached file %s", data_split_name, cached_features_file)
        torch.save(examples, cached_examples_file)
        torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    logger.info("Coverting to Tensors....")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_masks for f in features], dtype=torch.float)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if is_training:
        all_value_types = torch.tensor([f.value_types for f in features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_positions for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_positions for f in features], dtype=torch.long)
        all_domains = torch.tensor([f.domains for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_value_types,
                                all_start_positions, all_end_positions, all_domains)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long) # [0,1,...,999]
        dataset = TensorDataset(all_input_ids,       # batch[0]
                                all_attention_masks, # batch[1]
                                all_token_type_ids,  # batch[2]
                                all_example_index)   # batch[3]
    logger.info("Done")
    # Create slot ids
    slots_input = get_slot_input_ids(tokenizer, args.dataset_version, args.use_pattern)
    slots_input_ids = torch.tensor(slots_input[0], dtype=torch.long, device=args.device)
    slots_attention_masks = torch.tensor(slots_input[1], dtype=torch.long, device=args.device)
    if output_examples:
        return dataset, examples, features, (slots_input_ids, slots_attention_masks)
    return dataset, (slots_input_ids, slots_attention_masks)


def main():
    # Get args
    args = set_config()

    # Set up outputdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up devices
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(n_gpu=args.n_gpu, seed=args.seed)
    logger.warning("Process device: %s, n_gpu: %s, 16-bits training: %s", args.device, args.n_gpu, args.fp16)
    logger.info("Training/evaluation parameters %s", args)
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    
    if args.use_pattern:
        args.num_cls_labels = 30522
    else:
        args.num_cls_labels = 3

    # # Create config, tokenizer
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=args.num_cls_labels)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ANSWER_SEP]', '[SYS]', '[USER]']})

    
    model = SlotAttentionNetwork.from_pretrained(args.model_name_or_path,
                                                 from_tf=bool('.ckpt' in args.model_name_or_path),
                                                 config=config,
                                                 cache_dir=args.cache_dir if args.cache_dir else None,
                                                 args=args)

    ### Freeze BERT's weights ###
    if args.freeze_model:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False
    ### Freeze BERT's weights ###
    model.to(args.device)
    
    ### Create train, eval and test datasets ###
    train_dataloader, eval_dataloader, test_dataloader = None, None, None
    eval_examples, eval_features = None, None
    test_examples, test_features = None, None
    if args.do_train:
        train_dataset, slot_input_ids = load_dataset(args,
                                                     tokenizer, 
                                                     input_file=args.train_file,
                                                     data_split_name="train", 
                                                     is_training=True,
                                                     output_examples=False)
        # slot_input_ids = [i.repeat(args.n_gpu, 1, 1) for i in slot_input_ids]
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset)
        # Train DataLoaders
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    if args.do_eval:
        eval_dataset, eval_examples, eval_features, slot_input_ids = load_dataset(args, 
                                                                                tokenizer, 
                                                                                input_file=args.validation_file,
                                                                                data_split_name="eval",
                                                                                is_training=False,
                                                                                output_examples=True)
        # print("print eval features", [print(features[0][k]) for k in features[0] ])
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        # eval DataLoaders
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    if args.do_predict:
        test_dataset, test_examples, test_features, slot_input_ids = load_dataset(args, 
                                                                                tokenizer, 
                                                                                input_file=args.predict_file,
                                                                                data_split_name="test",
                                                                                is_training=False,
                                                                                output_examples=True)
        args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        test_sampler = SequentialSampler(test_dataset)
        # eval DataLoaders
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    ### Create train, eval and test datasets ###
    
    # Prepare the slot inputs (30,5)
    slot_input_ids = [i.repeat(args.n_gpu, 1, 1) for i in slot_input_ids]

    # Compute update steps 
    if args.max_steps > 0 and args.do_train is True: 
        args.t_total = args.max_steps
        args.num_train_epochs = (args.max_steps //
                                (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    elif args.do_train is True:
        args.t_total = (len(train_dataloader) //
                        args.gradient_accumulation_steps * args.num_train_epochs)
    else:
        args.t_total = 0
    # Set up optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_prob * args.t_total,
                                                num_training_steps=args.t_total)

    # Build trainer
    trainer = SlotAttentionNetworkTrainer(model=model,
                                          tokenizer=tokenizer,
                                          args=args,
                                          train_dataset=train_dataloader,
                                          eval_dataset=eval_dataloader,
                                          test_dataset=test_dataloader,
                                          eval_examples=eval_examples,
                                          eval_features=eval_features,
                                          test_examples=test_examples,
                                          test_features=test_features,
                                          compute_metrics=None,
                                          slot_input_ids=slot_input_ids,
                                          optimizer=optimizer,
                                          scheduler=scheduler)
    
    
    # Training
    if args.do_train:
        trainer.train()
        
    # Evaluation
    # results = {}
    # if args.do_eval:
    #     checkpoints = [args.output_dir]
    #     tb_writer = None
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
    #         checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    #         tb_writer = SummaryWriter(args.tensorboard_name)

    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)

    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         config = BertConfig.from_pretrained(checkpoint, num_labels=args.num_cls_labels)
    #         model = SlotAttentionNetwork.from_pretrained(checkpoint, config=config, args=args)
    #         model.to(args.device)
    #         result = trainer.evaluate(model, mode="eval", prefix=global_step)

    #         if args.eval_all_checkpoints:
    #             global_step = int(global_step)
    #             tb_writer.add_scalar('joint_acc', result[0], global_step)
    #             tb_writer.add_scalar('slot_acc', result[1], global_step)
    #             tb_writer.add_scalar('cls_acc', result[2], global_step)
    #             tb_writer.add_scalar('max_acc', result[3], global_step)
    #         results[global_step] = result
    #     if args.eval_all_checkpoints:
    #         tb_writer.close()

    #logger.info("Results: {}".format(results))
        

if __name__ == "__main__":
    main()
