"""Trainer for slot attention netowrk."""
import argparse
import logging
import os
import random
import glob
import timeit
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup,
                          get_constant_schedule_with_warmup)
from sa_model import SlotAttention
from sa_utils import (read_multiwoz_examples, convert_examples_to_features, get_slot_input_ids, multiwoz_evaluate,
                      RawResult, get_slots)
from sa_config import set_config
from models.slot_attention import SlotAttentionNetwork

from my_utils import set_seed, tensor_to_list

from transformers.file_utils import is_apex_available

logger = logging.getLogger(__name__)


# install apex from https://www.github.com/nvidia/apex to use fp16 training
# if multi-GPUs training
if is_apex_available():
    from apex import amp


def convert_id_to_value_type(inp):
    """Maaping token id into value type.

    This is neccessary when using `use_pattern`.
    The token ids of `none`, `no` and `span` will be
    convert into value type {0,1,2,3}. 
    0 is for `none` and ``

    value type   token ids (pattern)    token_ids (no pattern)   
    `none`     ==  `3904`            == `0`
    `dontcare` ==  `2053`            == `1`
    `span`     ==  `8487`            == `2`
    ``         == any other token id == `3`
    Args:
      inp: 1D tensor
    """
    # assert inp[0] == int
    #assert torch.is_tensor(inp[0]) == False
    MAPPING = {3904: 0, 2053:1, 8487:2}
    #print("input tensor", inp)
    inp = tensor_to_list(inp)
    #print("after to list", inp)
    return [MAPPING[v] if v in MAPPING else 3 for v in inp]



class SlotAttentionNetworkTrainer:
    """Construct Trainer for easily training and evalaution."""
    def __init__(self,
                 model,
                 tokenizer,
                 args,
                 train_dataset,
                 eval_dataset,
                 test_dataset,
                 eval_examples,
                 eval_features,
                 test_examples,
                 test_features,
                 compute_metrics, 
                 slot_input_ids,
                 optimizer,
                 scheduler):
        super(SlotAttentionNetworkTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataloader = train_dataset
        self.eval_dataloader = eval_dataset # none
        self.test_dataloader = test_dataset
        
        self.eval_examples = eval_examples
        self.eval_features = eval_features
        self.test_examples = test_examples
        self.test_features = test_features

        self.compute_metrics = compute_metrics # none
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.slot_input_ids = slot_input_ids# tensor with shape (30,5)

        self.num_cls_labels = self.args.num_cls_labels
        # logger.info("logger can be called!!!!!")


    def training_step(self, model, inputs):
        """Perform one training step."""
        model.train()
        # loss is: alpha * slot_loss + beta * (start_loss + end_loss)
        loss, slot_loss, start_loss, end_loss = self.compute_loss(model, inputs)

        # mean() to average on multi-gpu parallel training
        if self.args.n_gpu > 1:
            loss = loss.mean()
            slot_loss = slot_loss.mean()
            start_loss = start_loss.mean()
            end_loss = end_loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return (loss, slot_loss, start_loss, end_loss)


    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss of slot gate classification and span position."""
        # nn.CrossEntropyLoss: use logit as input
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        slot_logits, start_logits, end_logits = model(**inputs)
        

        value_types = inputs["value_types"]       # (batch_size, 30): value type is 0 (none), 1 (dontcare) or 2 (span).
        start_positions =inputs["start_positions"]# (batch_size, 30): position index or -1
        end_positions = inputs["end_positions"]   # (batch_size, 30): position index or -1

        ## Loss ###
        slot_loss = loss_fn(slot_logits.view(-1, self.num_cls_labels), value_types.view(-1))
        # Positions losses: logits as (batch_size*num_slot, seq_len) and label as (batch_size*num_slot)
        start_loss = loss_fn(start_logits.view(-1, start_logits.size(-1)), start_positions.view(-1))
        end_loss = loss_fn(end_logits.view(-1, start_logits.size(-1)), end_positions.view(-1))
        # Compute final loss
        # loss = (self.args.cls_lambda * slot_loss + \
        #        self.args.cls_lambda * (start_loss + end_loss))
        loss = self.args.ans_lambda * (start_loss+end_loss) + self.args.cls_lambda * slot_loss
        return (loss, slot_loss, start_loss, end_loss)
        

    def _wrap_model(self, model, training=True):
        """Wrap model in singel, multiple or not GPU(s) training."""
        # Mixed precision training with apex
        if self.args.fp16 and training is True:
            self.model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)
        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)
        
        return model


    def train(self):
        """Train Slot Attention Network."""
        # For the sake of using `args`
        args = self.args
        tb_writer = SummaryWriter(args.tensorboard_name)
        # multi-gpu training setting
        self.model = self._wrap_model(model=self.model)
        ### Logging training ###
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataloader.dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.t_total)

        global_step = 0
        train_loss, logging_loss, eval_best_acc = 0.0, 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        set_seed(n_gpu=args.n_gpu, seed=args.seed)
        
        ### Train with N epochs ###
        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[0],       # (batch_size, seq_len)
                          'attention_mask': batch[1],  # (batch_size, seq_len)
                          'token_type_ids': batch[2],  # (batch_size, seq_len) 
                          'value_types': batch[3],     # (batch_size, 30): value type is 0 (none), 1 (dontcare) or 2 (span).
                          'start_positions': batch[4], # (batch_size, 30): position index or -1
                          'end_positions': batch[5],   # (batch_size, 30): position index or -1
                          'slot_input_ids': self.slot_input_ids, # slot_input_ids has shape (30, 5)
                          "use_pattern": self.args.use_pattern }
                # loss: alpha * slot_loss + beta * (start_loss + end_loss)
                (loss, slot_loss, start_loss, end_loss) = self.training_step(self.model, inputs)

                loss_msg = (f"loss:{loss:.2f},  slot loss:{slot_loss:.2f}, "
                            f"start loss:{start_loss:.2f}, end loss:{end_loss:.2f}, "
                            f"eval acc:{eval_best_acc:.2f}")
                epoch_iterator.set_description(loss_msg, refresh=False)
                train_loss += loss.item()

                ### Optimizer step ###
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                        #logger.info("global step reach eval step!")
                        if args.evaluate_during_training:
                            joint_acc, slot_acc, cls_acc, max_acc = evaluate(args, 
                                                                             self.model,
                                                                             mode="eval",
                                                                             prefix=str(global_step))
                            tb_writer.add_scalar('joint_acc', joint_acc, global_step)
                            tb_writer.add_scalar('slot_acc', slot_acc, global_step)
                            tb_writer.add_scalar('cls_acc', cls_acc, global_step)
                            tb_writer.add_scalar('max_acc', max_acc, global_step)

                            logger.info(f"joint acc: {joint_acc} at global step: {global_step}")
                            logger.info(f"slot acc: {slot_acc} at global step: {global_step}")
                            logger.info(f"cls acc: {cls_acc} at global step: {global_step}")
                            logger.info(f"max acc: {max_acc} at global step: {global_step}")
                            eval_best_acc = max(eval_best_acc, joint_acc)

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        #logger.info("global step reach logging step!")
                        tb_writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (train_loss - logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar('slot_loss', slot_loss, global_step)
                        tb_writer.add_scalar('start_loss', start_loss, global_step)
                        tb_writer.add_scalar('end_loss', end_loss, global_step)
                        
                        logging_loss = train_loss

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        #logger.info("global step reach save step!")
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                ### Optimizer step ###

                # Stop inner loop
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            # Stop outer loop
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

            # Each epoch 
            if args.evaluate_during_training:
                # Evaluate on dev dataset
                joint_acc, slot_acc, cls_acc, max_acc = self.evaluate(mode="eval",
                                                                      prefix="eval dataset")
                tb_writer.add_scalar('joint_acc', joint_acc, global_step)
                tb_writer.add_scalar('slot_acc', slot_acc, global_step)
                tb_writer.add_scalar('cls_acc', cls_acc, global_step)
                tb_writer.add_scalar('max_acc', max_acc, global_step)

                # Evaluate on test set
                joint_acc, slot_acc, cls_acc, max_acc = self.evaluate(mode="test",
                                                                      eval_dataloader=self.test_dataloader,
                                                                      prefix="test dataset")
                
        tb_writer.close()

        # Saving model, tokenizer and args
        logger.info("Saving model checkpoint to %s", self.args.output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(args.output_dir)
        self.tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        return global_step, train_loss / global_step


    def evaluate(self, model=None, mode=None, eval_dataloader=None, prefix=""):
        """Evaluate methods. It can be called during or after training.

        `evalaute()` requires `features` and `examples` for search span

        Args:
          model: `SlotAttentionNetwork` class. If model provided, it will be used for evaluation.
          mode: determine which features and examples to post-process.
          eval_dataloader: Used for evalaution. otherwise use `self.eval_dataloader`
          prefix: str. Added to logger information.
        
        """
        if model is not None:
            model = model
        else:
            model = self.model

        #
        assert mode in {"eval", "test"}
        if mode == "eval":
            features = self.eval_features
            examples = self.eval_examples
        elif mode == "test":
            features = self.test_features
            examples = self.test_examples

        # If not provided, use default eval dataset
        if eval_dataloader is None:
            eval_dataloader = self.eval_dataloader

        # Set up
        model.eval()
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataloader.dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        # all_results = {}
        all_results = []
        start_time = timeit.default_timer()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'slot_input_ids': self.slot_input_ids,
                          "use_pattern": self.args.use_pattern
                          }
                # 1D tensor contains indices of examples: 
                # Wth batch size 64, each `example_indices` is
                # [0,1,2,...,63], [64,65,..], [..., num_example-1]
                example_indices = batch[3] 
                outputs = model(**inputs)
            
            # print("example indices !!!", example_indices)
            for i, example_index in enumerate(example_indices):
                # print("example_index.item()", example_index.item())
                # print("features[example_index.item()]", self.features[example_index.item()]) 
                # Get feature by index
                feature = features[example_index.item()]

                # if `use_pattern`, outputs[0] is type_logits with shape (batch, 30, 30522)
                # outputs[0][i] is i-th type_logits with shape (30, 3)
                one_dim_slot_logits = outputs[0][i].argmax(dim=-1)

                if self.args.use_pattern:
                    #print("before",one_dim_slot_logits)
                    one_dim_slot_logits = convert_id_to_value_type(one_dim_slot_logits)
                    #print("after", one_dim_slot_logits)

                result = RawResult(unique_id=int(feature.unique_id),
                                   types=one_dim_slot_logits,
                                   start_logits=tensor_to_list(outputs[1][i]),
                                   end_logits=tensor_to_list(outputs[2][i]))
                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataloader.dataset))
        results = multiwoz_evaluate(all_results, examples, features, self.args.dataset_version, use_vn=self.args.use_vn)

        return results