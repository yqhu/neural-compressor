#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .utils import logger
from .utils.utility import singleton
from .experimental import Pruning as ExpPruning

@singleton
class Pruning(object):
    """This is base class of pruning object.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and pruning objectives (performance, memory footprint etc.).
       Pruning class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        pruning objective and related dataloaders etc.

    """

    def __init__(self, conf_fname):
        self.exp_pruner = ExpPruning(conf_fname)

    def on_epoch_begin(self, epoch):
        """ called on the begining of epochs"""
        self.exp_pruner.on_epoch_begin(epoch)

    def on_batch_begin(self, batch_id):
        """ called on the begining of batches"""
        self.exp_pruner.on_batch_begin(batch_id)

    def on_batch_end(self):
        """ called on the end of batches"""
        self.exp_pruner.on_batch_end()

    def on_epoch_end(self):
        """ called on the end of epochs"""
        self.exp_pruner.on_epoch_end()

    def __call__(self, model, q_dataloader=None, q_func=None, eval_dataloader=None,
                 eval_func=None):
        """The main entry point of pruning.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandotory.

           b) Partial yaml configuration: User specifies dataloaders used in calibration
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create lpot dataloader before calling this
              function.

              After that, User specifies fp32 "model", calibration dataset "q_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and quantized model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the quantized model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_dataloader parameters are mandotory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandotory.

        Args:
            model (object):                        For PyTorch model, it's torch.nn.model
                                                   instance.
            q_dataloader (generator):              Data loader for calibration. It is iterable
                                                   and should yield a tuple (input, label) for
                                                   calibration dataset containing label,
                                                   or yield (input, _) for label-free calibration
                                                   dataset. The input could be a object, list,
                                                   tuple or dict, depending on user implementation,
                                                   as well as it can be taken as model input.
            q_func (function, optional):           Training function for pruning.
                                                   This function takes "model" as input parameter
                                                   and executes entire training process with self
                                                   contained training hyper-parameters. If this
                                                   parameter specified, eval_dataloader parameter
                                                   plus metric defined in yaml, or eval_func
                                                   parameter should also be specified at same time.
            eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                                   and should yield a tuple of (input, label).
                                                   The input could be a object, list, tuple or
                                                   dict, depending on user implementation,
                                                   as well as it can be taken as model input.
                                                   The label should be able to take as input of
                                                   supported metrics. If this parameter is
                                                   not None, user needs to specify pre-defined
                                                   evaluation metrics through configuration file
                                                   and should set "eval_func" paramter as None.
                                                   Tuner will combine model, eval_dataloader
                                                   and pre-defined metrics to run evaluation
                                                   process.
            eval_func (function, optional):        The evaluation function provided by user.
                                                   This function takes model as parameter,
                                                   and evaluation dataset and metrics should be
                                                   encapsulated in this function implementation
                                                   and outputs a higher-is-better accuracy scalar
                                                   value.

                                                   The pseudo code should be something like:

                                                   def eval_func(model):
                                                        input, label = dataloader()
                                                        output = model(input)
                                                        accuracy = metric(output, label)
                                                        return accuracy

        Returns:
            pruned model: best pruned model found, otherwise return None

        """
        logger.warning('This API is going to be deprecated, please import '
            'lpot.experimental.Pruning, set the attributes about '
            'dataloader and metric, then use new __call__ method')
        self.exp_pruner.model = model
        if q_dataloader or eval_dataloader or eval_func:
            logger.warning('Pruning will do nothing with param '
                           'q_dataloader/eval_dataloader/eval_func')
        self.exp_pruner.q_func = q_func 
        return self.exp_pruner()

