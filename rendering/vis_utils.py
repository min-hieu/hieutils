import numpy as np
import matplotlib.pyplot as plt
import mcubes
import torch
import fresnelvis as fresnelvis


def th2np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

def plot_gaussians(gaussians):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lim = 0.7
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    N = gaussians.shape[0]
    cmap = plt.get_cmap("jet")

    multiplier = 0.2
    for i, g in enumerate(gaussians):
        mu, R, eival = g[:3], g[3:12], g[13:]
        '''
        eivec = eivec.reshape(3,3)
        sigma = eivec @ np.diag(eival) @ eivec.T
        sigma_inv = np.linalg.inv(sigma)
        eigval, R = np.linalg.eigh(sigma_inv)
        '''

        a, b, c = multiplier * np.sqrt(eigval)
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones_like(u), np.cos(v))

        coord = np.stack((x,y,z), axis=-1)
        coord = np.einsum('ij,uvj->uvi', R, coord) # apply rot on all coord
        x = coord[...,0]
        y = coord[...,1]
        z = coord[...,2]

        ax.plot_surface(x,y,z, rstride=4, cstride=4, color='gray')

    return fig

def plot(pc, lim=0.7):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if not isinstance(pc, list):
        pc = [pc]
    for p in pc:
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    return fig


def make_grid(bb_min=[0, 0, 0], bb_max=[1, 1, 1], shape=[64, 64, 64], flatten=True):
    coords = []
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    if type(shape) is int:
        shape = np.array([shape] * bb_min.shape[0])
    for i, si in enumerate(shape):
        coord = np.linspace(bb_min[i], bb_max[i], si)
        coords.append(coord)
    grid = np.stack(np.meshgrid(*coords, sparse=False), axis=-1)
    if flatten:
        grid = grid.reshape(-1, grid.shape[-1])
    return grid


def grid2mesh(grid, thresh=0, smooth=False, bbmin=-1, bbmax=1):
    if smooth:
        grid = mcubes.smooth(grid)

    verts, faces = mcubes.marching_cubes(grid, thresh)
    # verts = verts[:,[2,0,1]]
    verts = verts / (grid.shape[0] - 1)
    verts = verts * (bbmax - bbmin) + bbmin
    faces = faces.astype(int)
    return verts, faces


def render_grid(
    grid,
    thresh=0.0,
    shapes=(64, 64, 64),
    camera_kwargs=dict(
        camPos=np.array([2, -2, -2]),
        camLookat=np.array([0.0, 0.0, 0.0]),
        camUp=np.array([1, 0, 0]),
        camHeight=2,
        resolution=(512, 512),
        samples=16,
    ),
):
    grid = th2np(grid)
    grid = grid.reshape(shapes)
    verts, faces = grid2mesh(grid, thresh)
    img = fresnelvis.renderMeshCloud(mesh={"vert": verts, "face": faces}, **camera_kwargs)
    return img
