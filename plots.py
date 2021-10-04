import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import torch



def scatter(X, labels=None, ax=None, colors=None, **kwargs):
    '''
    plot scatter points or scatter rectangles
    dim X = nb of points per image x dim of space
    '''
    
    ax = ax or plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    
    if X.size(1) == 2:
        if labels is None:
            ax.scatter(X[:,0].cpu().data.numpy(), X[:,1].cpu().data.numpy(), facecolor='k',
                    edgecolor=[0.2, 0.2, 0.2], **kwargs)
            return None
        else:
            ulabels = np.sort(np.unique(labels.cpu().numpy()))
            colors = cm.rainbow(np.linspace(0, 1, len(ulabels))) \
                    if colors is None else colors
            for (l, c) in zip(ulabels, colors):
                ax.scatter(X[labels==l,0].cpu().data.numpy(), X[labels==l,1].cpu().data.numpy(), color=c,
                        edgecolor=c*0.6, **kwargs)
            return ulabels, colors
        
    elif X.size(1)==4:
        R = rect_for_plots(X)
        
        if labels is None:
            for i in range(X.size(0)):
                ax.add_patch(
                            Rectangle((R[i, 0].cpu().data.numpy(), R[i, 1].cpu().data.numpy()), 
                              R[i, 2].cpu().data.numpy(), R[i, 3].cpu().data.numpy(), 
                              fill=False, color='blue', linewidth=1.5, alpha=0.5
                                     )
                            )
    
            ax.axis('equal')
            return None
        else:
            ulabels = np.sort(np.unique(labels.cpu().numpy()))
            colors = cm.rainbow(np.linspace(0, 1, len(ulabels)))\
                    if colors is None else colors

            for (l, c) in zip(ulabels, colors):
                R_temp=R[torch.where(labels==l)[0]]

                #Put sample rectangles
                for i in range(R_temp.size(0)):
                    ax.add_patch(
                        Rectangle((R_temp[i, 0].cpu().data.numpy(), R_temp[i, 1].cpu().data.numpy()), 
                                  R_temp[i, 2].cpu().data.numpy(), R_temp[i, 3].cpu().data.numpy(), 
                                  fill=False, color=c, linestyle ='-.', linewidth=1, alpha=0.4
                                 )
                    )

            ax.axis('equal')
            return ulabels, colors


def rect_for_plots(rects):
    '''
    input: N x 4 in format (coordinates upper left) x (coordinates bottom right)
    '''
    
    '''
    w = (rects[...,2] - rects[...,0]).unsqueeze(1)
    h = (rects[...,1] - rects[...,3])
    x2 = (rects[...,1] - h).unsqueeze(1)
    h = h.unsqueeze(1)
    x1 = rects[..., 0].unsqueeze(1)
    R = torch.hstack((x1, x2, w, h))
    '''
    
    X = torch.hstack((rects[:, 0].unsqueeze(1), rects[:, 2].unsqueeze(1)))
    Y = torch.hstack((rects[:, 1].unsqueeze(1), rects[:, 3].unsqueeze(1)))
    x_min, _ = torch.min(X, 1)
    y_min, _ = torch.min(Y, 1)
    x_max, _ = torch.max(X, 1)
    y_max, _ = torch.max(Y, 1)
    w = x_max - x_min
    h = y_max - y_min
    R = torch.hstack(
            (x_min.unsqueeze(1),
            y_min.unsqueeze(1),
            w.unsqueeze(1),
            h.unsqueeze(1))
        )
    
    return R


    
def draw_ellipse(pos, cov, ax=None, **kwargs):
    if type(pos) != np.ndarray:
        pos = to_numpy(poxs)
    if type(cov) != np.ndarray:
        cov = to_numpy(cov)
    ax = ax or plt.gca()
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 6):
        ax.add_patch(Ellipse(pos, nsig*width, nsig*height, angle,
            alpha=0.5/nsig, **kwargs))

def scatter_mog(X, labels, mu, cov, ax=None, colors=None):
    ax = ax or plt.gca()
    
    ulabels, colors = scatter(X, labels=labels, ax=ax, colors=colors, zorder=10)
    
    if X.size(1)==2:
        for i, l in enumerate(ulabels):
            draw_ellipse(mu[l].cpu().detach().numpy(), cov[l].cpu().detach().numpy(), ax=ax, fc=colors[i])

    else:

        for (l, c) in zip(ulabels, colors):
            #Add ellipses for upper left and  bottom right corners

            draw_ellipse(mu[l,0:2].cpu().detach().numpy(), 
                         cov[l, 0:2, 0:2].cpu().detach().numpy(), 
                         ax=ax, fc=c
                        )
            draw_ellipse(mu[l,2:4].cpu().detach().numpy(),
                         cov[l, 2:4, 2:4].cpu().detach().numpy(), 
                         ax=ax, fc=c
                        )
            
            #ax.axis('equal')
            
        mu_plot = rect_for_plots(mu)    
        for (l, c) in zip(ulabels, colors):
            #Put predicted mean box for this mixture component
            ax.add_patch(
                Rectangle((mu_plot[l, 0].cpu().data.numpy(), mu_plot[l, 1].cpu().data.numpy()), 
                            mu_plot[l, 2].cpu().data.numpy(), mu_plot[l, 3].cpu().data.numpy(), 
                            fill=False, color='black', linestyle ='-', linewidth=1.5, alpha=0.6
                            )
            )    
    
            #ax.axis('equal')