import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser

from backbone.two_clip.model_two import clip_two
from backbone.two_clip_open_clip.factory import clip_two_open_clip
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
import math
from transformers import CLIPProcessor
import json
from utils.data_manager import DataManager
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, roc_auc_score
import os
import re
import open_clip
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EPSILON = 1e-8

def add_special_args(parser:ArgumentParser) -> ArgumentParser: 
    parser.add_argument("--m_desp", type=bool, default=None, help="Whether to use multiple text descriptions")
    parser.add_argument("--lambd",  type=int, default=None, help='hyperparameter of image to image similarity') 
    parser.add_argument("--beta",  type=int, default=None, help='hyperparameter of image to text similarity')
    parser.add_argument("--alpha",  type=int, default=None, help='hyperparameter of ce loss')    
    parser.add_argument("--gama",  type=int, default=None, help='hyperparameter of text distance loss')
    parser.add_argument("--theta",  type=int, default=None, help='hyperparameter of text unchange loss')
    parser.add_argument("--phi",  type=int, default=None, help='hyperparameter of cur logits loss')
    parser.add_argument("--test_bs",  type=int, default=128, help='test batch size') 
    parser.add_argument("--is_OOD_test", type=bool, default=None, help="Whether to do OOD testing")
    parser.add_argument("--T",  type=float, default=None, help='Temperature')
    parser.add_argument("--sim_threshold",  type=float, default=None, help='Control the distance between text embeddings')
    parser.add_argument("--visual_threshold",  type=int, default=None, help='Control the distribution between visual prototypes')
class CLIP_TWO_open_clip(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._m_desp = config.m_desp
        self._lambda = config.lambd
        self._beta = config.beta
        self._gama = config.gama
        self._theta = config.theta
        self._phi = config.phi
        self._alpha = config.alpha
        self.sim_threshold = config.sim_threshold
        self.visual_threshold = config.visual_threshold 
        self._is_openset_test = config.is_OOD_test
        self._T = config.T
        self._test_bs = config.test_bs
        self._init_cls = config.init_cls
        self._increment = config.increment
        self._til_record = []
        self._cil_record = []
        self._tid_record = []
        
        if self._is_openset_test:
            self._AUC_record = []
            self._FPR95_record = []

        if config.backbone == "ViT-B-16":
            # self._pretrained_weights_path = "/home/2021/wentao/Storage/tmp/CLIP_OOD_detection_CIL/pretrain_weights/clip_vit_base_patch16"
            self._pretrained_weights_path = "/home/2021/wentao/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-DataComp.XL-s13B-b90K/snapshots/d110532e8d4ff91c574ee60a342323f28468b287/open_clip_pytorch_model.bin"
            # self._pretrained_weights_path = "pretrain_weights/clip_vit_base_patch16"
        elif config.backbone == "ViT-B-16-SigLIP":
            self._pretrained_weights_path = "/home/2021/wentao/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP/snapshots/41f575766f40e752fdd1383e9565b7f02388c1c4/open_clip_pytorch_model.bin"
        ## 
        # self._clip_process = CLIPProcessor.from_pretrained(self._pretrained_weights_path)
        if self._m_desp:
            if config.dataset == 'cifar100_i2t'or config.dataset == 'cifar100_i2t_few_shot':
                desp_json = 'datasets/cifar100_prompts_base.json'
            elif config.dataset == 'imagenetr_i2t':
                desp_json = 'datasets/I2T_Imagenet_r.json'
            elif config.dataset == 'imagenet100_i2t' or config.dataset == 'imagenet100_i2t_new':
                desp_json = 'datasets/imagenet100.json'
            elif config.dataset == 'mini_imagenet100_i2t':
                desp_json = 'datasets/mini_imagenet.json'
            elif config.dataset == 'skin8_i2t':
                desp_json = 'datasets/skin8_prompt.json'
        else:
            if config.dataset == 'cifar100_i2t'or config.dataset == 'cifar100_i2t_few_shot':
                desp_json = 'datasets/cifar100_one_prompt.json'
            elif config.dataset == 'imagenetr_i2t':
                desp_json = 'datasets/I2T_imagenet_r_one_prompt.json'
            elif config.dataset == 'skin8_i2t':
                desp_json = 'datasets/skin8_prompt_shot.json'
            elif config.dataset == 'skin40_i2t':
                desp_json = 'datasets/skin40_one_prompt.json'
            elif config.dataset == 'p12_i2t':    
                desp_json = 'datasets/p12_one_prompt.json'
            elif config.dataset == 'mini_imagenet100_i2t':    
                desp_json = 'datasets/mini_imagenet_one_prompt.json'
            
        id_class_desp = []
        # with open("datasets/cifar100_prompts_full.json") as f:
        with open(desp_json) as f:
            id_texts = json.load(f)
        # load description
        for i in range(len(id_texts[list(id_texts.keys())[0]])):
            id_class_desp.append([id_texts[label][i] for label in list(id_texts.keys())])

        self._id_text_tokens = {}
        self._tokenizer = open_clip.get_tokenizer(self._backbone)
        # tokenizer
        for i in range(len(id_class_desp)):
            self._id_text_tokens.update({i : self._tokenizer(id_class_desp[i])})
        
        # 每类只有一个prompt
        self.text_tokens = self._id_text_tokens[0] 
        self._logger.info('Applying CLIP_OOD_CIL (a class incremental method, test with {})'.format(self._incre_type))

        # initialize the model
        self._network = clip_two_open_clip(self._logger, self._backbone, self._pretrained_weights_path)

        if config.dataset == 'cifar100_i2t':
            OOD_dataset = 'cifar100_i2t_ood'
        elif config.dataset == 'cifar100_i2t_few_shot':
            OOD_dataset = 'cifar100_i2t_few_shot_ood'
        elif config.dataset == 'imagenetr_i2t':
            OOD_dataset = 'imagenetr_i2t_ood'
        elif config.dataset == 'skin40_i2t':
            OOD_dataset = 'skin40_i2t_ood'
        elif config.dataset == 'mini_imagenet100_i2t':
            OOD_dataset = 'mini_imagenet100_i2t_ood'

        self._data_manager_OOD = DataManager(logger, OOD_dataset, config.img_size, config.split_dataset, 
                                            config.shuffle, config.seed, config.init_cls, config.increment, config.use_valid) 

    def prepare_model(self):
        self._cur_task += 1
        # update_visual_encoder
        if self._cur_task > 0:
            self._network.update_visual_encoder()
        self._network = self._network.cuda()

    def prepare_task_data(self, data_manager_ID):
        
        self._cur_classes = data_manager_ID.get_task_size(self._cur_task)
        print("self._known_classes", self._known_classes)
        print("self._cur_classes", self._cur_classes)
        self._total_classes = self._known_classes + self._cur_classes
        

        
        self._train_dataset_ID = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        
        self._train_dataset_ID_prototype = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='test')

        self._train_dataset_OOD = self._data_manager_OOD.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='train', mode='train')
            
        self._test_dataset = data_manager_ID.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._test_dataset_fc = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager_ID.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._cur_task_test_samples_num = len(self._test_dataset)

        self._logger.info('Train dataset of ID size: {}'.format(len(self._train_dataset_ID)))
        self._logger.info('Train dataset of OOD size: {}'.format(len(self._train_dataset_OOD)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))
        self._logger.info('Test dataset of current task size: {}'.format(len(self._test_dataset_fc)))
        

        self._train_loader_ID = DataLoader(self._train_dataset_ID, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._train_loader_prototype = DataLoader(self._train_dataset_ID_prototype, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._train_loader_OOD = DataLoader(self._train_dataset_OOD, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

        self._test_loader = DataLoader(self._test_dataset, batch_size=self._test_bs, shuffle=False, num_workers=self._num_workers)
        self._test_fc_loader = DataLoader(self._test_dataset_fc, batch_size=self._test_bs, shuffle=False, num_workers=self._num_workers)

        self._iters_per_epoch_lora =  math.ceil(len(self._train_dataset_ID)*1.0/self._batch_size)

        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._order = torch.tensor(data_manager_ID._class_order)

      

    def incremental_train(self):

        
        # train lora
        self._network.update_mlp(self._cur_classes)
        self._network = self._network.cuda()
        self._logger.info("Training current task-special visual lora and share lora with data of current task!")
        self._network.train_mode()
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        optimizer = self._get_optimizer( self._network.parameters(), self._config, False)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self._epochs*self._iters_per_epoch_lora)

        self._network = self._train_model(self._network, self._train_loader_ID, self.text_tokens, optimizer, scheduler,  test_loader=self._test_fc_loader,
                                            task_id=self._cur_task, epochs=self._epochs, note='_')
        
        self._network.test_mode()
        
        # cal prototype
        self._cal_image_prototype(self._network, self._train_loader_prototype, task_id=self._cur_task)

        # cal text features
        self._cal_text_feature(self._network, self.text_tokens, task_id=self._cur_task)

        # save lora and prototype
        self._save_checkpoint('seed{}_task{}_checkpoint.pkl'.format(self._seed, self._cur_task),
                self._network.cpu())

  
    def _train_model(self, model, train_loader, text_tokens, optimizer, scheduler,  test_loader=None,  task_id=None, epochs=100, note=''):
        
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        if note != '':
            note += '_'

        self._scaler = GradScaler()
        for epoch in range(epochs):
            model, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler, 
                                                    text_tokens=text_tokens, task_begin=task_begin, task_end=task_end, task_id=task_id)
            
            info = ('Task {}, Epoch {}/{} => '.format(task_id, epoch+1, epochs) + ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses))
             
            self._logger.info(info)
        
        cur_classes = self._order[task_begin : task_end]
        model.eval()
        with torch.no_grad():
            id_text_features = model.get_texts_feature(text_tokens[cur_classes].cuda())
            id_text_features /= id_text_features.norm(p=2, dim=-1, keepdim=True)
        test_acc = self._epoch_test(model, test_loader,  text_features=id_text_features, task_begin=task_begin, task_end=task_end, task_id=task_id)
        
        info = info + 'test_acc {:.3f}, '.format(test_acc) 
        self._logger.info(info)

        self._til_record.append(test_acc)
        self._logger.info("TIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._til_record)).format(*self._til_record) + ']')
        return model

    def _epoch_train(self, model, train_loader,  optimizer, scheduler, text_tokens=None, task_begin=None, task_end=None, task_id=None):
        # self._network.train_mode()
        losses = 0
        clip_losses = 0.
        text_distance_losses = 0.
        text_unchange_losses = 0.
        cur_logits_losses = 0.
        ce_losses = 0.

        correct = 0.
        total = 0
        cur_class_num = self._increment_steps[task_id]
        cur_pre_classes = self._order[:task_end]
        model.train()
        ood_loader = iter(self._train_loader_OOD)
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            B, _, _, _ = inputs.shape
            targets = targets - task_begin

            _, ood_inputs, ood_targets = next(ood_loader)
            ood_inputs, ood_targets = ood_inputs.cuda(), ood_targets.cuda()
            ood_targets = torch.tensor([cur_class_num]*len(ood_targets)).cuda()
            inputs = torch.cat((inputs, ood_inputs), dim=0)
            all_targets = torch.cat((targets, ood_targets), dim=0)
            # all_targets = targets
            with autocast():
                image_features_all, text_features = model(inputs, text_tokens[cur_pre_classes].cuda())  # forward(self, image_inputs, input_ids, attention_mask)
                image_logits = model.get_mlp(image_features_all)
            
            ce_loss = self._ce_loss(image_logits, all_targets)
            image_features = image_features_all[:B,:] / image_features_all[:B,:].norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  
            cur_classes_text_features = text_features[task_begin:task_end,:]
            logit_scale = model.logit_scale.exp()
            logits_per_image = torch.matmul(image_features.float(), cur_classes_text_features.float().t()) * logit_scale 
            clip_loss = self._clip_loss(logits_per_image, targets)   # [:,:cur_class_num]
            if task_id == 0:
                text_distance_loss = self._text_distance_loss(cur_classes_text_features, threshold=self.sim_threshold)
                loss = clip_loss + self._gama * text_distance_loss+ ce_loss*self._alpha
                # loss = clip_loss + self._gama * text_distance_loss
            else:
                old_classes_text_features = text_features[:task_begin,:]
                text_distance_loss = self._text_distance_loss(cur_classes_text_features, old_text=old_classes_text_features, threshold=self.sim_threshold)
                text_unchange_loss = self._text_cos_loss(text_features[:task_begin,:], self.pre_text_features)


                loss = clip_loss + self._gama * text_distance_loss + self._theta * text_unchange_loss  + ce_loss*self._alpha

            
            # loss = clip_loss           

            optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.step(optimizer)
            self._scaler.update()

            
            clip_losses += clip_loss.item()
            text_distance_losses += text_distance_loss.item()
            ce_losses += ce_loss.item()
            if task_id > 0:
                text_unchange_losses += text_unchange_loss.item()
                # cur_logits_losses += cur_logits_loss.item()

           
            _, predicted = (logits_per_image.max(1))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            losses += loss.item()
            
            if scheduler != None:
                scheduler.step()
        train_loss_acc = ['Loss', losses/len(train_loader),  'clip_loss', clip_losses/len(train_loader),  'text_distance_loss', text_distance_losses/len(train_loader),   'text_unchange_loss', text_unchange_losses/len(train_loader), 'ce_losses', ce_losses/len(train_loader), 'train_acc', correct / (total+EPSILON)*100]
        # train_loss_acc = ['Loss', losses/len(train_loader),  'clip_loss', clip_losses/len(train_loader),  'text_distance_loss', text_distance_losses/len(train_loader),   'text_unchange_loss', text_unchange_losses/len(train_loader), 'train_acc', correct / (total+EPSILON)*100]

        return model, train_loss_acc

    def _epoch_test(self, model, test_loader,  text_features=None, task_begin=None, task_end=None, task_id=None):
        # self._network.test_mode()
        cur_class_num = self._increment_steps[task_id]
        correct = 0.
        total = 0
        
        model.eval()
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets - task_begin
                with autocast():
                    image_features = model.get_images_feature(inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_per_image = torch.matmul(image_features.float(), text_features.float().t()) * logit_scale

                _, predicted = (logits_per_image.max(1))
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc =correct / (total+EPSILON)*100

        return test_acc
    
    def _cal_image_prototype(self, model, train_loader, task_id=None):
        task_begin = sum(self._increment_steps[:task_id])
        cur_num_classes = self._increment_steps[task_id]
        self.image_prototype = torch.zeros(cur_num_classes, 512).float().cuda()
        model.eval()
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets - task_begin
                with autocast():
                    image_features = model.get_images_feature(inputs)
                # image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                self.image_prototype.scatter_add_(0, targets.unsqueeze(1).expand(-1, image_features.size(1)), image_features.float())
            self.image_prototype /= self.image_prototype.norm(p=2, dim=-1, keepdim=True)

        if task_id == 0:
            self.learned_image_prototype = self.image_prototype
        else:
            self.learned_image_prototype = torch.cat((self.learned_image_prototype, self.image_prototype), dim=0)

    def _cal_text_feature(self, model, text_tokens, task_id=None):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        cur_pre_classes = self._order[:task_end]
        model.eval()
        with torch.no_grad():
            self.pre_text_features = model.get_texts_feature(text_tokens[cur_pre_classes].cuda())
            self.pre_text_features /= self.pre_text_features.norm(p=2, dim=-1, keepdim=True)

        self.pre_text_embeddings = {}
        for i in range(task_id+1):
            task_begin = sum(self._increment_steps[:i])
            task_end = task_begin + self._increment_steps[i]
            self.pre_text_embeddings.update({i : self.pre_text_features[task_begin:task_end,:]})

    

    
    def eval_task(self):
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed
        
        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]): # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else: # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id:path}

        chk_paths = seed_checkpoint_paths[self._seed]


        pre_tasks_classes = torch.tensor([sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()
        if self._is_openset_test  and self._cur_task < self._nb_tasks-1:
            self._test_loader = self._openset_test_loader

        for cur_task in range(len(chk_paths)):
            chk_name = chk_paths[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)
            self.image_prototype = tmp_checkpoint['image_prototype']
            self.image_prototype = self.image_prototype.cuda()
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_text_features = self.pre_text_features[task_begin:task_end,:]  

            self._network.eval()
            idx = 0
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    # B, _, _, _ = inputs.shape
                    with autocast():
                        image_features = self._network.get_images_feature(inputs)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    img_img_sim = torch.matmul(image_features.float(), self.image_prototype.t())
                    img_text_sim = torch.matmul(image_features.float(), cur_task_text_features.t())
                    logits_per_image = self._lambda * img_img_sim + img_text_sim * self._beta
                    ood_scores_per_image, predicted = torch.max(logits_per_image, dim=1, keepdim=True)


                   
                    if idx == 1:
                        cur_task_ood_scores =  ood_scores_per_image
                        cur_task_preds = predicted
                        cur_targets = targets
                        cur_task_logits = logits_per_image
                    else:
                        cur_task_ood_scores = torch.cat((cur_task_ood_scores, ood_scores_per_image), dim=0)
                        cur_task_preds = torch.cat((cur_task_preds, predicted), dim=0)
                        cur_targets = torch.cat((cur_targets, targets), dim=0)
                        cur_task_logits = torch.cat((cur_task_logits, logits_per_image), dim=0)
                    
            if cur_task == 0:
                all_tasks_ood_scores = cur_task_ood_scores
                all_tasks_preds = cur_task_preds
                all_task_logits = cur_task_logits
            else:
                all_tasks_ood_scores = torch.cat((all_tasks_ood_scores, cur_task_ood_scores), dim=1)
                all_tasks_preds = torch.cat((all_tasks_preds, cur_task_preds), dim=1)
                all_task_logits = torch.cat((all_task_logits, cur_task_logits), dim=1)
            
            self._network = self._network.cpu()

        task_id_per_image = torch.argmax(all_tasks_ood_scores, dim=1)
        task_pred_per_image = all_tasks_preds[range(len(all_tasks_ood_scores)), task_id_per_image]
        all_tasks_preds = pre_tasks_classes[task_id_per_image] + task_pred_per_image
        cur_all_preds = all_tasks_preds[:self._cur_task_test_samples_num]
        cur_total = cur_targets[:self._cur_task_test_samples_num]
        if self._eval_metric == "acc":
            total = cur_total.size(0)
            correct = cur_all_preds.eq(cur_total).sum().item()
            t_id_correct = task_id_per_image[:self._cur_task_test_samples_num].eq(cur_total//self._increment).sum().item()
            t_id_acc = t_id_correct/total*100
            acc = correct/total*100
            self._logger.info("After training the {}th task, the accuracy of the test set: {}".format(len(chk_paths)-1, acc))
            self._cil_record.append(acc)
            self._tid_record.append(t_id_acc)
            self._logger.info("Task ID: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._tid_record)).format(*self._tid_record) + ']')
        elif self._eval_metric == "mcr":
            cm = confusion_matrix(cur_total.cpu(), cur_all_preds.cpu())
            right_of_class = np.diag(cm)
            num_of_class = cm.sum(axis=1)
            task_size = cm.shape[0]
            mcr = np.around((right_of_class*100 / (num_of_class+1e-8)).sum() / task_size, decimals=2)
            self._logger.info("After training the {}th task, the mean class recall of the test set: {}".format(len(chk_paths)-1, mcr))
            self._cil_record.append(mcr)
        else:
            assert self._eval_metric != "mcr" and self._eval_metric != "acc", "Please enter the correct eval metric (mcr or acc)!"  
        
        if self._is_openset_test and self._cur_task < self._nb_tasks-1:
            labels_list = [1]*self._cur_task_test_samples_num
            labels_list.extend([0]*(len(self._openset_test_dataset) - self._cur_task_test_samples_num))
            scores = all_task_logits
            if self._T==0 or self._T == None:
                scores_softmax = scores
            else:
                scores_softmax = torch.softmax(scores.float()/self._T, dim=1)
            max_scores = torch.max(scores_softmax, dim=1)[0]
            scores_list = max_scores.tolist()
            rocauc = roc_auc_score(labels_list, scores_list)
            fpr, tpr, _ = roc_curve(labels_list, scores_list)
            fpr95_idx = np.where(tpr>=0.95)[0]
            fpr95 = fpr[fpr95_idx[0]]
            self._AUC_record.append(rocauc*100)
            self._FPR95_record.append(fpr95*100)
            
            self._logger.info("AUC curve of all stages is [\t" + ("{:2.2f}\t"*len(self._AUC_record)).format(*self._AUC_record) + ']')
            self._logger.info("FPR95 curve of all stages is [\t" + ("{:2.2f}\t"*len(self._FPR95_record)).format(*self._FPR95_record) + ']')

        self._logger.info("CIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._cil_record)).format(*self._cil_record) + ']')
        
    # 利用已经学习过类别的text embeddings和当前类别的visual embeddings一起构造OOD
    def _construct_ood(self, visual, text):
        B, _ = visual.shape
        N, _ = text.shape
        random_idx = torch.randint(0, N, (B,))
        selected_text = text[random_idx]

        fake_ood = visual + selected_text

        return fake_ood
        


    # 该loss用于算图文匹配
    def _clip_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
    
    def _ce_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    # 以下loss用于保证旧的text embeddings不发生变化的   
    def _text_cos_loss(self, pred, targets):
        # return (1 - F.cosine_similarity(pred, targets)).sum()
        return (1 - F.cosine_similarity(pred, targets)).mean()

    
    

    # 该loss是用于控制新学的text embeddings与旧的text embeddings之间的距离，以及新学的text embeddings与新学的text embeddings之间的距离
    def _text_distance_loss(self, new_text, old_text=None, threshold=0.7):
        # new_text = new_text / new_text.norm(p=2, dim=-1, keepdim=True)
        if old_text is not None: 
            # old_text = old_text / old_text.norm(p=2, dim=-1, keepdim=True)   
            text_embeddings = torch.cat((new_text, old_text), dim=0)
            text_sim = torch.matmul(new_text, text_embeddings.t()).fill_diagonal_(0)
            text_sim = text_sim - threshold
            text_sim = F.relu(text_sim)
        else:
            text_sim = torch.matmul(new_text, new_text.t()).fill_diagonal_(0)
            text_sim = text_sim - threshold
            text_sim = F.relu(text_sim)
        
        return text_sim.sum()/(text_sim.size(0)*text_sim.size(1)-text_sim.size(0))
        # return text_sim.sum()

        
    def after_task(self):
        self._known_classes = self._total_classes
        


    def _save_checkpoint(self, filename, model=None):
        save_path = os.path.join(self._logdir, filename)

        # save lora 
        my_state_dict = model.state_dict()
        # model_state_dict = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k and 'image_features' in k}
        # model_state_dict = {k: my_state_dict[k] for k in my_state_dict if 'image_features' in k}
        model_state_dict = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k }


        # save model config
        model_state_dict.update({'config':self._config.get_parameters_dict()})

        # save current task
        model_state_dict.update({'task_id':self._cur_task})

        # save prototype
        model_state_dict.update({'image_prototype':self.image_prototype})

        # save text features
        model_state_dict.update({'pre_text_features':self.pre_text_features})

        torch.save(model_state_dict, save_path)
        self._logger.info('checkpoint saved at: {}'.format(save_path))


    def store_samples(self):
        pass
        

    def eval_cil_task(self):
        task_id = self._cur_task
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed
        
        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]): # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else: # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id:path}

        chk_paths = seed_checkpoint_paths[self._seed]
        
        # load self.pre_text_features
        self.pre_text_features = torch.load(os.path.join(chks_path, chk_paths[task_id]))['pre_text_features']
        self.pre_text_features = self.pre_text_features.cuda()



        pre_tasks_classes = torch.tensor([sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()

        if self._is_openset_test  and self._cur_task < self._nb_tasks-1:
            self._test_loader = self._openset_test_loader
        ##
        for cur_task in range(task_id+1):
            chk_name = chk_paths[cur_task]
            class_num = self._increment_steps[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)
            self.image_prototype = tmp_checkpoint['image_prototype']
            self.image_prototype = self.image_prototype.cuda()            
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_text_features = self.pre_text_features[task_begin:task_end,:]  
            


            self._network.eval()
            self._network._training_mode = 'test'
            idx = 0
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        image_features = self._network.get_images_feature(inputs)

                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    img_img_sim = torch.matmul(image_features.float(), self.image_prototype.t())
                    img_text_sim = torch.matmul(image_features.float(), cur_task_text_features.t())
                    logits_per_image = self._lambda * img_img_sim + img_text_sim * self._beta
                    ood_scores_per_image, predicted = torch.max(logits_per_image, dim=1, keepdim=True)

                    if idx == 1:
                        cur_task_ood_scores =  ood_scores_per_image
                        cur_task_preds = predicted
                        cur_targets = targets
                        cur_task_logits = logits_per_image
                    else:
                        cur_task_ood_scores = torch.cat((cur_task_ood_scores, ood_scores_per_image), dim=0)
                        cur_task_preds = torch.cat((cur_task_preds, predicted), dim=0)
                        cur_targets = torch.cat((cur_targets, targets), dim=0)
                        cur_task_logits = torch.cat((cur_task_logits, logits_per_image), dim=0)
            if cur_task == 0:
                all_tasks_ood_scores = cur_task_ood_scores
                all_tasks_preds = cur_task_preds
                all_task_logits = cur_task_logits
            else:
                all_tasks_ood_scores = torch.cat((all_tasks_ood_scores, cur_task_ood_scores), dim=1)
                all_tasks_preds = torch.cat((all_tasks_preds, cur_task_preds), dim=1)
                all_task_logits = torch.cat((all_task_logits, cur_task_logits), dim=1)

            self._network = self._network.cpu()
        
        task_id_per_image = torch.argmax(all_tasks_ood_scores, dim=1)
        task_pred_per_image = all_tasks_preds[range(len(all_tasks_ood_scores)), task_id_per_image]
        all_tasks_preds = pre_tasks_classes[task_id_per_image] + task_pred_per_image
        cur_all_preds = all_tasks_preds[:self._cur_task_test_samples_num]
        cur_total = cur_targets[:self._cur_task_test_samples_num]
        if self._eval_metric == "acc":
            total = cur_total.size(0)
            correct = cur_all_preds.eq(cur_total).sum().item()
            t_id_correct = task_id_per_image[:self._cur_task_test_samples_num].eq(cur_total//self._increment).sum().item()
            t_id_acc = t_id_correct/total*100
            acc = correct/total*100
            self._logger.info("After training the {}th task, the accuracy of the test set: {}".format(task_id, acc))
            self._cil_record.append(acc)
            self._tid_record.append(t_id_acc)
            self._logger.info("Task ID: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._tid_record)).format(*self._tid_record) + ']')
        elif self._eval_metric == "mcr":
            cm = confusion_matrix(cur_total.cpu(), cur_all_preds.cpu())
            right_of_class = np.diag(cm)
            num_of_class = cm.sum(axis=1)
            task_size = cm.shape[0]
            mcr = np.around((right_of_class*100 / (num_of_class+1e-8)).sum() / task_size, decimals=2)
            self._logger.info("After training the {}th task, the mean class recall of the test set: {}".format(task_id, mcr))
            self._cil_record.append(mcr)
        else:
            assert self._eval_metric != "mcr" and self._eval_metric != "acc", "Please enter the correct eval metric (mcr or acc)!"  
        
        if self._is_openset_test and self._cur_task < self._nb_tasks-1:
            labels_list = [1]*self._cur_task_test_samples_num
            labels_list.extend([0]*(len(self._openset_test_dataset) - self._cur_task_test_samples_num))
            scores = all_task_logits
            if self._T==0 or self._T == None:
                scores_softmax = scores
            else:
                scores_softmax = torch.softmax(scores.float()/self._T, dim=1)
            max_scores = torch.max(scores_softmax, dim=1)[0]
            scores_list = max_scores.tolist()
            rocauc = roc_auc_score(labels_list, scores_list)
            fpr, tpr, thresholds = roc_curve(labels_list, scores_list)
            fpr95_idx = np.where(tpr>=0.95)[0]
            fpr95 = fpr[fpr95_idx[0]]
            self._AUC_record.append(rocauc*100)
            self._FPR95_record.append(fpr95*100)
            
            self._logger.info("AUC curve of all stages is [\t" + ("{:2.2f}\t"*len(self._AUC_record)).format(*self._AUC_record) + ']')
            self._logger.info("FPR95 curve of all stages is [\t" + ("{:2.2f}\t"*len(self._FPR95_record)).format(*self._FPR95_record) + ']')
        
        self._logger.info("CIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._cil_record)).format(*self._cil_record) + ']')
        





  