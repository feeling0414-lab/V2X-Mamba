# -*- coding: utf-8 -*-
# Author: Quanhao Li

"""
Dataset class for early fusion
"""
import random
import math
from collections import OrderedDict

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import v2xvit.data_utils.datasets
import v2xvit.data_utils.post_processor as post_processor
from v2xvit.utils import box_utils

from v2xvit.data_utils.datasets import intermediate_fusion_dataset
from v2xvit.data_utils.augmentor.data_augmentor import DataAugmentor
from v2xvit.data_utils.pre_processor import build_preprocessor
from v2xvit.hypes_yaml.yaml_utils import load_yaml
from v2xvit.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum

#新增加
from v2xvit.utils.transformation_utils import x1_to_x2
import v2xvit.utils.pcd_utils as pcd_utils
from v2xvit.utils.transformation_utils import tfm_to_pose
from v2xvit.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from v2xvit.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from v2xvit.utils.transformation_utils import x_to_world


import numpy as np
import torch
import math
import copy
import time

import v2xvit.data_utils.post_processor as post_processor
from v2xvit.data_utils.datasets import basedataset
from v2xvit.data_utils.pre_processor import build_preprocessor
from v2xvit.hypes_yaml.yaml_utils import load_yaml
from v2xvit.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from v2xvit.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from v2xvit.utils.pose_utils import add_noise_data_dict
from v2xvit.utils import box_utils


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

#Dair-V2X
class IntermediateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self). \
            __init__(params, visualize, train)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False
        '''
        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = read_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']

        else:
            self.box_align = False
        '''

        # print("OPV2V dataset initialized! \
        #         {} samples totally!".format(self.len_record[-1]))

    def __getitem__(self, idx):

        base_data_dict = self.retrieve_base_data(idx)

        base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break

        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        processed_features = []
        object_stack = []
        object_id_stack = []
        too_far = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        projected_lidar_clean_list = []
        cav_id_list = []

        spatial_correction_matrix = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue

            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])  # 6dof pose
            cav_id_list.append(cav_id)


        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]

            # obj:
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose,
                ego_lidar_pose_clean,
                idx)

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            #增加
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])

            processed_features.append(
                selected_cav_processed['processed_features'])
            if self.kd_flag:
                projected_lidar_clean_list.append(
                    selected_cav_processed['projected_lidar_clean'])

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        ######################################################

        ############ for disconet ###########
        if self.kd_flag:
            stack_lidar_np = np.vstack(projected_lidar_clean_list)
            stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                                  self.params['preprocess'][
                                                      'cav_lidar_range'])
            stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)

        merged_feature_dict = self.merge_features_to_dict(processed_features)
        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
            spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)


        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'spatial_correction_matrix': spatial_correction_matrix,
             'lidar_poses': lidar_poses})

        if self.kd_flag:
            processed_data_dict['ego'].update({'teacher_processed_lidar':
                                                   stack_feature_processed})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})

        processed_data_dict['ego'].update({'sample_idx': idx,
                                           'cav_id_list': cav_id_list})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose, ego_pose_clean, idx):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        idx: int,
            debug use.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)  # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                     ego_pose_clean)

        # retrieve objects under ego coordinates
        # this is used to generate accurate GT bounding box.
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                                                     ego_pose_clean)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        # x,y,z in ego space
        projected_lidar = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        if self.kd_flag:
            lidar_np_clean = copy.deepcopy(lidar_np)

        if self.proj_first:
            lidar_np[:, :3] = projected_lidar

        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': projected_lidar,
             'processed_features': processed_lidar,
             'transformation_matrix': transformation_matrix,
             'transformation_matrix_clean': transformation_matrix_clean})

        if self.kd_flag:
            projected_lidar_clean = \
                box_utils.project_points_by_matrix_torch(lidar_np_clean[:, :3],
                                                         transformation_matrix_clean)
            lidar_np_clean[:, :3] = projected_lidar_clean
            lidar_np_clean = mask_points_by_range(lidar_np_clean,
                                                  self.params['preprocess'][
                                                      'cav_lidar_range'])
            selected_cav_processed.update(
                {"projected_lidar_clean": lidar_np_clean}
            )

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)  # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []


        if self.kd_flag:
            teacher_processed_lidar_list = []
        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses'])  # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            processed_lidar_list.append(
                ego_dict['processed_lidar'])  # different cav_num, ego_dict['processed_lidar'] is list.
            record_len.append(ego_dict['cav_num'])

            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix'])

            if self.kd_flag:
                teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)

        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)  # coords 增加了 batch_id 维度，[x, y, z] -> [id, x, y, z]
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list))

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'lidar_pose_clean': lidar_pose_clean,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'lidar_pose': lidar_pose})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        if self.kd_flag:
            teacher_processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(teacher_processed_lidar_list)
            output_dict['ego'].update({'teacher_processed_lidar': teacher_processed_lidar_torch_dict})

        if self.params['preprocess']['core_method'] == 'SpVoxelPreprocessor' and \
                (output_dict['ego']['processed_lidar']['voxel_coords'][:,
                 0].max().int().item() + 1) != record_len.sum().int().item():
            return None

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        # transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch,
                                   'transformation_matrix_clean':
                                       transformation_matrix_clean_torch, })

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })

        # output_dict['ego'].update({'veh_frame_id': batch[0]['ego']['veh_frame_id']})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1))  # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


class IntermediateFusionDatasetDAIR(IntermediateFusionDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        self.max_cav = 2
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        if self.train:
            split_dir = params['train_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['root_dir']
        self.split_info = load_json(split_dir)
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        frame_infomation = self.split_info[idx]
        system_error_offset = frame_infomation["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() # veh-side
        data[0]['ego'] = True
        data[1] = OrderedDict() # inf-side
        data[1]['ego'] = False
        veh_frame_id = frame_infomation['vehicle_image_path'][-10:-4]
        inf_frame_id = frame_infomation['infrastructure_pointcloud_path'][-10:-4]

        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_infomation['cooperative_label_path']))
        # print(data[0]['params']['vehicles'])
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)
        spatial_correction_matrix = np.eye(4)
        data[0]['params']['spatial_correction_matrix'] = spatial_correction_matrix

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_infomation["vehicle_pointcloud_path"]))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]

        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_infomation['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        data[1]['params']['vehicles'] = [] # we only load cooperative label in vehicle side

        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))

        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)
       #增加
        data[1]['params']['spatial_correction_matrix'] = spatial_correction_matrix
        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_infomation["infrastructure_pointcloud_path"]))
        return data

    def __len__(self):
        return len(self.split_info)

    ### rewrite generate_object_center ###
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
