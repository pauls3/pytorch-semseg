import os
import torch
import argparse, glob
import numpy as np
import scipy.misc as misc

from PIL import Image
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    allfiles = [args.img_path]
    if os.path.isdir(args.img_path):
        allfiles = files_in_subdirs(args.img_path)
    # img = misc.imread(args.img_path)

    # Setup image
    # if not calc_pred_quality:
        # print("Read Input Image from : {}, model {}, inp.sz:".format(args.img_path,model_name_shrt), orig_size)
    from tqdm import tqdm_notebook as tqdm
    # else:
        # tqdm = lambda *i, **kwargs: i[0]  # pylint:disable=invalid-name

    data_loader = get_loader(args.dataset)
    loader = data_loader(root="", is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes

    outdir = args.out_path
    outp_is_dir = max(outdir.find('.jpg'), outdir.find('.png')) < 0

    num_files = len(allfiles)
    all_hists = np.zeros((num_files,n_classes),dtype=np.int32)
    all_qual = np.zeros((num_files),dtype=np.float)
    res_idx = 0
    
    img = misc.imread(allfiles[0])
    out_size = img.shape[:-1]

    if args.inp_dim == None:
        img = misc.imread(allfiles[0])
        orig_size = img.shape[:-1]
    else:
        orig_size = [int(dim) for dim in args.inp_dim.split("x")]
        orig_size = [orig_size[1],orig_size[0]]


    if outp_is_dir:
        outdir += '/'
    if not os.path.exists(os.path.dirname(outdir)):
        os.makedirs(os.path.dirname(outdir))
    for f in tqdm(allfiles, "Calculating predictions..."):
        outname = outdir
        if outp_is_dir:
            outfile0 = os.path.basename(f).replace('.jpg','.png')
            # if args.check_options:
            #     outfile0 = args.version + "_" + str(args.img_norm) + outfile0
            outname = os.path.join(os.path.dirname(outdir), outfile0)
        img = misc.imread(f)

        # resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")
        resized_img = misc.imresize(img, (orig_size[0], orig_size[1]), interp="bicubic")
        img = resized_img

        if model_name in ["pspnet", "icnet", "icnetBN"]:
            # uint8 with RGB mode, resize width and height which are odd numbers
            img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
        else:
            img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= loader.mean
        if args.img_norm:
            img = img.astype(float) / 255.0

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        # Setup Model
        model_dict = {"arch": model_name}
        model = get_model(model_dict, n_classes, version=args.dataset)
        state = convert_state_dict(torch.load(args.model_path)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)

        images = img.to(device)
        outputs = model(images)
        

        if args.dcrf:
            unary = outputs.data.cpu().numpy()
            unary = np.squeeze(unary, 0)
            unary = -np.log(unary)
            unary = unary.transpose(2, 1, 0)
            w, h, c = unary.shape
            unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
            unary = np.ascontiguousarray(unary)

            resized_img = np.ascontiguousarray(resized_img)

            d = dcrf.DenseCRF2D(w, h, loader.n_classes)
            d.setUnaryEnergy(unary)
            d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

            q = d.inference(50)
            mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
            decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
            dcrf_path = args.out_path[:-4] + "_drf.png"
            misc.imsave(dcrf_path, decoded_crf)
            print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        pred = np.array(pred, dtype=np.uint8)
        # setting this temporarily for cityscapes output
        # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0).astype('uint8')
        if model_name in ["pspnet", "icnet", "icnetBN"] or args.resize_pred:
            pred = pred.astype(np.float32)
            # float32 with F mode, resize back to orig_size
            pred = misc.imresize(pred, out_size, "nearest", mode="F")
        # pred = misc.imresize(pred, out_size, "nearest", mode="L")
        decoded = loader.decode_segmap(pred)
        # Image.new("L", )
        # print("Classes found: ", np.unique(pred))
        misc.imsave(outname, decoded)
        #
        # print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--inp_dim",
        nargs="?",
        type=str,
        default=None,
        help="Fix input/output dimensions (e.g. 1920x1080); default: use dimensions of first test image",
    )
    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--resize_pred",
        dest="resize_pred",
        action="store_true",
        help="Resize image to original input image size",
    )
    parser.set_defaults(dcrf=False, resize_pred=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )
    args = parser.parse_args()
    test(args)
