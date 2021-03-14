import torch
import torchvision
import os
import glob
from torch.nn import functional as F
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def normMap(map):
    map_min = map.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0]
    map_max = map.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    map = (map-map_min)/(map_max-map_min)
    return map

def mixAndCat(catImg,map,img):
    mix = 0.8*map+0.2*img.mean(dim=1,keepdim=True)
    return torch.cat((catImg,mix),dim=0)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--ids',type=int,nargs="*")
    parser.add_argument('--inds',type=str)
    parser.add_argument('--show_inds', action='store_true')
    parser.add_argument('--plot_id',type=str)
    parser.add_argument('--nrow',type=int)
    parser.add_argument('--gradcam_id',type=str)
    parser.add_argument('--main_id',type=str)
    parser.add_argument('--high_res_id',type=str)
    args = parser.parse_args()

    if not os.path.exists("../vis/market/"):
        if not os.path.exists("../vis/"):
            os.makedirs("../vis/")
        os.makedirs("../vis/market")

    paths = glob.glob("../results/market/simMaps*_model{}*.pt".format(args.main_id))
    batch_inds = list(map(lambda x:int(os.path.basename(x).split("_")[0].replace("simMaps","")),paths))

    revIds = {}
    if args.inds is not None:
        inds = args.inds.split(";")
        for i in range(len(inds)):
            revIds[args.ids[i]] = i
            inds[i] = inds[i].split(",")
            for j in range(len(inds[i])):
                inds[i][j] = int(inds[i][j])
        print(inds)
    else:
        inds = None

    grid = None
    indCount = {}
    cmPlasma = plt.get_cmap('plasma')

    for ind in batch_inds:

        if args.ids is None:
            grid = None

        if len(glob.glob("../results/market/simMaps{}_model{}*.pt".format(ind,args.main_id))) == 0:
            simMaps = torch.load("../results/market/simMaps{}.pt".format(ind),map_location="cpu")
            norms = torch.load("../results/market/norm{}.pt".format(ind),map_location="cpu")
            imgs = torch.load("../results/market/img{}.pt".format(ind),map_location="cpu")
            ids = torch.load("../results/market/ids{}.pt".format(ind),map_location="cpu")
        else:
            simMaps = torch.load(glob.glob("../results/market/simMaps{}_model{}*.pt".format(ind,args.main_id))[0],map_location="cpu")
            norms = torch.load(glob.glob("../results/market/norm{}_model{}*.pt".format(ind,args.main_id))[0],map_location="cpu")

        if len(glob.glob("../results/market/img{}_model{}*.pt".format(ind,args.main_id))) == 0:
            imgs = torch.load("../results/market/img{}.pt".format(ind,args.main_id),map_location="cpu")
            ids = torch.load("../results/market/ids{}.pt".format(ind,args.main_id),map_location="cpu")
        else:
            imgs = torch.load(glob.glob("../results/market/img{}_model{}*.pt".format(ind,args.main_id))[0],map_location="cpu")
            ids = torch.load(glob.glob("../results/market/ids{}_model{}*.pt".format(ind,args.main_id))[0],map_location="cpu")

        simMaps = simMaps[:,:3]

        if args.ids is None or len(list(set(args.ids).intersection(set(ids)))) > 0:

            print("../results/market/simMaps{}.pt".format(ind))

            norms = normMap(norms)
            simMaps = normMap(simMaps)
            simMaps = norms*simMaps
            simMaps = normMap(simMaps)

            if not args.gradcam_id is None:
                print("../results/market/gradcam{}_model{}*_trial*.pt".format(ind,args.gradcam_id))
                gradcams = torch.load(glob.glob("../results/market/gradcam{}_model{}*_trial*.pt".format(ind,args.gradcam_id))[0],map_location="cpu")
                gradcams_pp = torch.load(glob.glob("../results/market/gradcam_pp{}_model{}*_trial*.pt".format(ind,args.gradcam_id))[0],map_location="cpu")
                guidedbacks = torch.load(glob.glob("../results/market/guidedback{}_model{}*_trial*.pt".format(ind,args.gradcam_id))[0],map_location="cpu")

                if len(glob.glob("../results/market/simMaps{}_model{}*.pt".format(ind,args.high_res_id))) ==0:
                    simMaps_hr = torch.load("../results/market/simMaps{}.pt".format(ind),map_location="cpu")
                    norms_hr = torch.load("../results/market/norm{}.pt".format(ind),map_location="cpu")
                else:
                    print(glob.glob("../results/market/simMaps{}_model{}*.pt".format(ind,args.high_res_id))[0])
                    simMaps_hr = torch.load(glob.glob("../results/market/simMaps{}_model{}*.pt".format(ind,args.high_res_id))[0],map_location="cpu")
                    norms_hr = torch.load(glob.glob("../results/market/norm{}_model{}*.pt".format(ind,args.high_res_id))[0],map_location="cpu")

                norms_hr = normMap(norms_hr)
                simMaps_hr = normMap(simMaps_hr)
                simMaps_hr = norms_hr*simMaps_hr
                simMaps_hr = normMap(simMaps_hr)

            for j in range(len(imgs)):

                if args.ids is None or ids[j] in args.ids:

                    matchingId = list(set(args.ids).intersection(set([ids[j]])))[0] if not args.ids is None else None

                    if (args.ids is None) or (ids[j] == matchingId):

                        if ids[j] not in indCount:
                            indCount[matchingId] = 0

                        if (args.ids is None) or (indCount[matchingId] in inds[revIds[matchingId]]):
                            img = imgs[j:j+1]

                            if args.show_inds:
                                imgPIL = Image.fromarray((255*img[0].permute(1,2,0).numpy()).astype("uint8"))
                                imgDraw = ImageDraw.Draw(imgPIL)
                                imgDraw.rectangle([(0,0), (220, 40)],fill="white")
                                imgDraw.text((0,0), str(ids[j])+" ",fill=(0,0,0))
                                img = torch.tensor(np.array(imgPIL)).permute(2,0,1).unsqueeze(0).float()/255

                            if grid is None:
                                grid = img
                            else:
                                grid = torch.cat((grid,img),dim=0)

                            if not args.gradcam_id is None:
                                #Guided
                                guided= normMap(guidedbacks[j:j+1])
                                guided = torch.abs(guided - guided.mean())
                                guided = guided.mean(dim=1,keepdims=True)
                                guided= normMap(guided)
                                guided = F.max_pool2d(guided,2)

                                guided = guided*F.interpolate(gradcams_pp[j:j+1],size=(guided.size(-2),guided.size(-1)))

                                guided = torch.tensor(cmPlasma(guided[0,0].numpy())[:,:,:3]).float()
                                guided = guided.permute(2,0,1).unsqueeze(0)
                                guided = F.interpolate(guided,size=(imgs.size(-2),imgs.size(-1)))
                                grid = mixAndCat(grid,guided,img)

                                #Gradcam
                                gradcam = torch.tensor(cmPlasma(gradcams[j,0])[:,:,:3]).float()
                                gradcam = gradcam.permute(2,0,1).unsqueeze(0)
                                gradcam = F.interpolate(gradcam,size=(imgs.size(-2),imgs.size(-1)))
                                grid = mixAndCat(grid,gradcam,img)

                                #Gradcam++
                                gradcam_pp = torch.tensor(cmPlasma(gradcams_pp[j,0])[:,:,:3]).float()
                                gradcam_pp = gradcam_pp.permute(2,0,1).unsqueeze(0)
                                gradcam_pp = F.interpolate(gradcam_pp,size=(imgs.size(-2),imgs.size(-1)))
                                grid = mixAndCat(grid,gradcam_pp,img)

                                #Norm
                                norm = torch.tensor(cmPlasma(norms[j,0])[:,:,:3]).float()
                                norm = norm.permute(2,0,1).unsqueeze(0)
                                norm = F.interpolate(norm,size=(imgs.size(-2),imgs.size(-1)),mode="bilinear")
                                grid = mixAndCat(grid,norm,img)

                            #AttMaps
                            fact_sim = imgs.size(-1)//simMaps.size(-1)
                            simMap = F.interpolate(simMaps[j:j+1],scale_factor=fact_sim,mode="bilinear")
                            grid = mixAndCat(grid,simMap,img)

                            if not args.gradcam_id is None:
                                #AttMaps
                                fact_sim = imgs.size(-1)//simMaps_hr.size(-1)
                                simMap_hr = F.interpolate(simMaps_hr[j:j+1],scale_factor=fact_sim)
                                grid = mixAndCat(grid,simMap_hr,img)

                        indCount[ids[j]] += 1

        if args.ids is None:
            torchvision.utils.save_image(grid,"../vis/market/attMaps{}_{}.png".format(i,args.plot_id))

    outPath = "../vis/market/attMaps_{}.png".format(args.plot_id)
    torchvision.utils.save_image(grid,outPath,nrow=args.nrow)
    os.system("convert  -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))
