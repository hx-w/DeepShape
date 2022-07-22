'''
for opened mesh parameterization
'''
from copy import deepcopy
from functools import reduce
import numpy_indexed as npi
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from utils import *

'''
return (
    inn_verts: list,
    bnd_verts: list,
    bnd_length: float
)
'''
def split_bnd_inn(msh: Trimesh) -> tuple:
    bnd_edges = npi.difference(msh.edges_unique, msh.face_adjacency_edges)

    bnd_verts = np.array([*bnd_edges[0]])
    bnd_edges = np.delete(bnd_edges, [0], axis=0)
    bnd_length = mesh_vert_dist(msh, *bnd_verts[:2])

    success = True
    while success:
        success = False
        last = bnd_verts[-1]
        for idx, edge in enumerate(bnd_edges):
            if last == edge[0]:
                success = True
                last = edge[1]
            elif last == edge[1]:
                success = True
                last = edge[0]
            if success:
                bnd_verts = np.append(bnd_verts, last)
                bnd_edges = np.delete(bnd_edges, [idx], axis=0)
                bnd_length += mesh_vert_dist(msh, *bnd_verts[-2:])
                break

    inn_verts = npi.difference(msh.face_adjacency_edges.flatten(), bnd_verts)
    return (inn_verts, bnd_verts, bnd_length)

'''
parameterize mesh boundary to square
return f_B: list
'''
def mapping_boundary(msh: Trimesh, bnd_verts: list, bnd_length: float, scale: float=2.) -> list:
    last_v = bnd_verts[0]
    accumed = 0.

    bnd_verts = bnd_verts[1:]
    f_B = []

    for bnd_v in bnd_verts:
        old_ratio = accumed / bnd_length
        accumed += mesh_vert_dist(msh, last_v, bnd_v)
        ratio = accumed / bnd_length
        flag = -reduce(
            lambda x, y: x * (
                1 if ((y - old_ratio) * (y - ratio)) > 0 
                else -y
            ),
            [0.25, 0.5, 0.75],
            1
        )
        if flag > 0:
            ratio = flag
        vpos = (0., 0.)
        if ratio < 0.25:
            vpos = (-(scale / 2) + scale * (ratio / 0.25), -scale / 2)
        elif ratio < 0.5:
            vpos = (scale / 2,  -(scale / 2) + scale * ((ratio - 0.25) / 0.25))
        elif ratio < 0.75:
            vpos = ((scale / 2) - scale * ((ratio - 0.5) / 0.25), scale / 2)
        else:
            vpos = (-scale / 2, (scale / 2) - scale * ((ratio - 0.75) / 0.25))

        f_B.append(np.append(vpos, 0.))
        last_v = bnd_v

    return f_B


'''
*** initial weights ***
'''
def initialize_weights(msh: Trimesh, inn_verts: list, bnd_verts: list) -> csc_matrix:
    # sub function
    def weights_for_edge(edge: list) -> float:
        adj_list_s = msh.vertex_neighbors[edge[0]]
        adj_list_b = msh.vertex_neighbors[edge[1]]
        adj_vts = npi.intersection(adj_list_s, adj_list_b)
        # assert len(adj_vts) == 2, 'not a manifold'
        # compute cotangent weight of edge
        ang1 = mesh_vert_angle(msh, adj_vts[0], *edge)
        ang2 = mesh_vert_angle(msh, adj_vts[1], *edge)
        _w = (math_cot(ang1) + math_cot(ang2)) / 2
        return -_w

    # sparse matrix index
    sp_row = np.array([], dtype=int)
    sp_col = np.array([], dtype=int)
    sp_data = np.array([], dtype=float)
    mtx_diag = np.zeros(len(msh.vertices))
    # generate
    _weights = list(map(weights_for_edge, msh.face_adjacency_edges))
    # update diag
    for idx, edge in enumerate(msh.face_adjacency_edges):
        mtx_diag[edge[0]] += -_weights[idx]
        mtx_diag[edge[1]] += -_weights[idx]
    
    # transpose indices
    _indices = msh.face_adjacency_edges.T
    sp_row = np.hstack([sp_row, _indices[0], _indices[1]])
    sp_col = np.hstack([sp_col, _indices[1], _indices[0]])
    sp_data = np.hstack([sp_data, _weights, _weights])

    # handle diag sparse index
    # all vertices in msh with order {INNER, BOUND}
    sp_diag_index = np.append(inn_verts, bnd_verts)
    sp_row = np.hstack([sp_row, sp_diag_index])
    sp_col = np.hstack([sp_col, sp_diag_index])
    sp_diag_data = [mtx_diag[v] for v in sp_diag_index]
    sp_data = np.hstack([sp_data, sp_diag_data])

    sp_weights = csc_matrix((sp_data, (sp_row, sp_col)), dtype=float)
    return sp_weights


'''
split sp_weights with sp_weights_II and sp_weights_IB
and solve equation:
    sp_weights_II * f_I = -sp_weights_IB * f_B
'''
def solve_equation(sp_weights: csc_matrix, f_B: list, inn_verts: list, bnd_verts: list) -> list:
    _mid = sp_weights[inn_verts, ...]
    sp_weights_II = _mid[..., inn_verts]
    sp_weights_IB = _mid[..., bnd_verts]

    assert sp_weights_IB.shape[1] == len(f_B), 'L_IB * f_B illegal'

    f_I = spsolve(sp_weights_II, -sp_weights_IB * f_B)
    return f_I


'''
build param mesh by inverse mapping
assume Z=0 in param mesh
'''
def build_param_mesh(msh: Trimesh, inn_verts: list, bnd_verts: list, f_I: list, f_B: list) -> Trimesh:
    len_inn, len_bnd = len(inn_verts), len(bnd_verts)
    param_bnd_verts = [v + len_inn for v in range(len_bnd)]
    inv_mapping = dict(zip(bnd_verts, param_bnd_verts))
    param_inn_verts = [v for v in range(len_inn)]
    inv_mapping.update(zip(inn_verts, param_inn_verts))
    param_tot = np.append(f_I, f_B, axis=0)

    param_mesh = Trimesh(
        vertices=[param_tot[inv_mapping[i]] for i in range(len_inn + len_bnd)],
        faces=deepcopy(msh.faces)
    )
    return param_mesh

'''
build str mesh by sample vertices on param mesh
'''
def build_str_mesh(uns_mesh: Trimesh, param_mesh: Trimesh, sample_nums: int=50, scale: float=2.) -> Trimesh:
    assert sample_nums > 2, 'sample_nums too small'
    # flatten numpy elements to list will accelerate in cycle
    flt_faces = param_mesh.faces.tolist()
    flt_area_faces = param_mesh.area_faces.tolist()
    flt_vertices = param_mesh.vertices.tolist()
    flt_raw_faces = [[flt_vertices[tri[0]], flt_vertices[tri[1]], flt_vertices[tri[2]]] for tri in flt_faces]

    str_mesh = Trimesh()
    # return (triangle: np.array, area: float)
    def which_trias_in(pos: tuple) -> tuple:
        for idx, tri in enumerate(flt_faces):
            if flt_area_faces[idx] - 0.0 < 1e-6:
                continue
            _s = mesh_vec_cross(flt_raw_faces[idx][0], flt_raw_faces[idx][2], pos)
            _t = mesh_vec_cross(flt_raw_faces[idx][1], flt_raw_faces[idx][0], pos)
            if (_s < 0) != (_t < 0) and _s != 0 and _t != 0:
                continue
            _d = mesh_vec_cross(flt_raw_faces[idx][2], flt_raw_faces[idx][1], pos)
            if _d == 0 or (_d < 0) == (_s + _t <= 0):
                return tri, flt_area_faces[idx]
        assert False, 'not found'

    for ir in tqdm(range(sample_nums), desc='remeshing'):
        for ic in range(sample_nums):
            _x = scale * ir / (sample_nums - 1) - scale / 2
            _y = scale * ic / (sample_nums - 1) - scale / 2
            spot_trias, tot_area = which_trias_in((_x, _y, 0))
            vi_area = mesh_trias_area(
                (_x, _y, 0),
                param_mesh.vertices[spot_trias[1]],
                param_mesh.vertices[spot_trias[2]],
            )
            vj_area = mesh_trias_area(
                param_mesh.vertices[spot_trias[0]],
                (_x, _y, 0),
                param_mesh.vertices[spot_trias[2]],
            )
            vk_area = mesh_trias_area(
                param_mesh.vertices[spot_trias[0]],
                param_mesh.vertices[spot_trias[1]],
                (_x, _y, 0),
            )
            pnt = (vi_area * uns_mesh.vertices[spot_trias[0]] +
                vj_area * uns_mesh.vertices[spot_trias[1]] +
                vk_area * uns_mesh.vertices[spot_trias[2]]) / tot_area

            str_mesh.vertices = np.vstack([str_mesh.vertices, pnt])

            if ir > 0 and ic > 0:
                idx = ir * sample_nums + ic
                str_mesh.faces = np.vstack([
                    str_mesh.faces,
                    [idx, idx - sample_nums, idx - 1],
                    [idx - 1, idx - sample_nums, idx - sample_nums - 1]
                ])

    str_mesh.fill_holes()
    str_mesh.fix_normals()
    return str_mesh
