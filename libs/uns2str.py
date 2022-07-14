'''
@brief convert unstructured mesh to structured mesh
@author: hx-w
'''

import trimesh

class Uns2Str:
    '''
    Use parameterization method
    '''
    def __init__(self) -> None:
        self._uns_mesh = trimesh.Trimesh()
        self._str_mesh = trimesh.Trimesh()
        pass

    def read(self, path: str) -> bool:
        try:
            self._uns_mesh = trimesh.load_mesh(path)
            return True
        except:
            return False

    def save(self, path: str) -> bool:
        pass


if __name__ == '__main__':
    mtd = Uns2Str()
    mtd.read('static/unsmesh/simple_tooth.obj')
