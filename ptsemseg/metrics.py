# # Adapted from score written by wkentaro
# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

# import numpy as np


# class runningScore(object):
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.confusion_matrix = np.zeros((n_classes, n_classes))

#     def _fast_hist(self, label_true, label_pred, n_class):
#         mask = (label_true >= 0) & (label_true < n_class)
#         hist = np.bincount(
#             n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
#         ).reshape(n_class, n_class)
#         return hist

#     def update(self, label_trues, label_preds):
#         for lt, lp in zip(label_trues, label_preds):
#             self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

#     def get_scores(self):
#         """Returns accuracy score evaluation result.
#             - overall accuracy
#             - mean accuracy
#             - mean IU
#             - fwavacc
#         """
#         hist = self.confusion_matrix
#         acc = np.diag(hist).sum() / hist.sum()
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#         acc_cls = np.nanmean(acc_cls)
#         iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#         mean_iu = np.nanmean(iu)
#         freq = hist.sum(axis=1) / hist.sum()
#         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#         cls_iu = dict(zip(range(self.n_classes), iu))

#         return (
#             {
#                 "Overall Acc: \t": acc,
#                 "Mean Acc : \t": acc_cls,
#                 "FreqW Acc : \t": fwavacc,
#                 "Mean IoU : \t": mean_iu,
#             },
#             cls_iu,
#         )

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


# class averageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrices = {}
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds, add_names=[]):
        for idx_z, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            if lp.shape[0] != lt.shape[0] or lp.shape[1] != lp.shape[1]:
                pl = []
                for i,d in enumerate(lt.shape):
                    p0 = d-lp.shape[i]
                    p1 = int(p0/2)
                    pl += [p1, p0-p1]
                lp0 = np.pad(lp,[(pl[0],pl[1]),(pl[2],pl[3])],mode='constant')
                lp = lp0
            single_mat = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            if len(add_names) > idx_z:
                self.confusion_matrices[add_names[idx_z]] = single_mat
            self.confusion_matrix += single_mat

    def get_confmats(self):
        return self.confusion_matrices

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count