import argparse
import os
import time

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils, infrence_utils
from v2xvit.data_utils.datasets import build_dataset
from v2xvit.visualization import vis_utils
from v2xvit.utils import eval_utils

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=12,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)



    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()


    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                   0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='result',width=2000,height=1000)

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        # vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

        view_control = vis.get_view_control()


    for i, batch_data in enumerate(data_loader):
        view_control.set_lookat([0,0,0]) #x,y,z

        view_control.set_zoom(0.12)

        # if i == 57 or i==103 or i ==115 or i ==205 or i ==232 or i ==261 or i==459 or i ==608:
        #     continue

        #if i%10!=0:
        #    continue
        print(i)
        # if i==3:
        #    continue
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor,output_dict = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                Lidar_map,pred_box_tensor, pred_score, gt_box_tensor,output_dict = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset)


            # elif opt.fusion_method == 'intermediate':
            #     pred_box_tensor, pred_score, gt_box_tensor,output_dict = \
            #         infrence_utils.inference_intermediate_fusion(batch_data,
            #                                                      model,
            #                                                      opencood_dataset)

            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            if pred_box_tensor == None:
                print("no pred_box")
                continue
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)



            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    Lidar_map_save_path = os.path.join(opt.model_dir, 'Lidar')

                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    heatmap_save_path = os.path.join(vis_save_path, '%05dheatmap.png' % i)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                infrence_utils.draw_Lidar_feature_map(Lidar_map,Lidar_map_save_path,i)
                #infrence_utils.draw_attention_mid(output_dict,vis_save_path,i)
                # infrence_utils.draw_heat_map(attention_map, vis_save_path,i)
                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)


            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if len(gt_o3d_box)==0:
                    continue
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')
                if len(pred_o3d_box)==0:
                    continue
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                vis.run()
                #time.sleep(5)

                vis_save_path = os.path.join(saved_path, 'vis')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                vis.capture_screen_image(vis_save_path)


    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
