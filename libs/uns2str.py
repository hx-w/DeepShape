'''
@brief convert unstructured mesh to structured mesh
@author: hx-w
'''

from typing import Tuple
import trimesh
from opened_param import *

class Uns2Str:
    '''
    Use parameterization method
    '''
    def __init__(self) -> None:
        pass

    def __read(self, path: str) -> Trimesh:
        try:
            _uns_mesh = trimesh.load_mesh(path)
            _uns_mesh.fill_holes()
            _uns_mesh.remove_duplicate_faces()
            return _uns_mesh
        except:
            return Trimesh()

    def read_convert(self, source_path: str, apply_obb: bool=False) -> trimesh.Trimesh:
        _uns_mesh = self.__read(source_path)
        return self.convert(_uns_mesh, apply_obb)

    def convert(self, uns_mesh: Trimesh, apply_obb: bool=False) -> Tuple[Trimesh, Trimesh]:
        inn_verts, bnd_verts, bnd_length = split_bnd_inn(uns_mesh)
        f_B = mapping_boundary(uns_mesh, bnd_verts, bnd_length)
        bnd_verts = bnd_verts[1:]
        sp_weights = initialize_weights(uns_mesh, inn_verts, bnd_verts)
        # important
        f_I = solve_equation(sp_weights, f_B, inn_verts, bnd_verts)
        # reconstruct
        param_mesh = build_param_mesh(uns_mesh, inn_verts, bnd_verts, f_I, f_B)
        str_mesh = build_str_mesh(uns_mesh, param_mesh, 100)

        if apply_obb: str_mesh.apply_obb()
        return str_mesh


if __name__ == '__main__':
    str_mesh = Uns2Str().read_convert('static/unsmesh/simple_tooth.obj', True)
    str_mesh.show()
