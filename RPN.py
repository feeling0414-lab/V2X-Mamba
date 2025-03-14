import torch.nn as nn
rpn_losses,proposal_list = self.rpn_head.forward_train(
    x,#输入网络的金字塔特征
    img_metas,#输入图象的参数
    gt_bboxes,#真实边界框

)
class RPNNetwork(nn.Module):
    def __init__(self,in_channels):
        super(RPNNetwork,self).__init__()
        self.Conv1=nn.Conv2d(in_channels,512,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.cls_score = nn.Conv2d(512,2*9,kernel_size=1)#9表示anchor值
        self.bbox_pred = nn.Conv2d(512,4*9,kernel_size=1)

    def forward(self,x,img_metas,gt,bboxes):
        x = self.relu(self.Conv1(x))
        rpn_cls_score = self.cls_score(x)
        rpn_bbox_score = self.bbox_pred(x)
        return rpn_cls_score,rpn_bbox_score

