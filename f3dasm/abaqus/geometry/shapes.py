'''
Created on 2020-10-15 09:36:46
Last modified on 2020-11-23 16:03:54

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (DEFORMABLE_BODY, THREE_D, ON, CLOCKWISE,
                             YZPLANE, XYPLANE, XZPLANE, FROM_SECTION)

# standard library
import copy

# TODO: refactor

# abstract object

class Shape(object):

    def __init__(self, name, material=None):
        self.name = name
        self.material = material

        # mesh definitions
        self.mesh_size = .02
        self.mesh_deviation_factor = .4
        self.mesh_min_size_factor = .4

    def change_mesh_definitions(self, **kwargs):
        # TODO: use general create mesh
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_inner_geometry(self, sketch, rve_info):
        '''
        Perform operations in main sketch.
        '''
        pass

    def create_part(self, sketch, rve_info):
        pass

    def creat_instance(self, model):
        return None


# particular shapes

# TODO: periodic sphere?
class Sphere(Shape):

    # TODO: add material

    def __init__(self, r, center=None, periodic=False, tol=1e-4, name='SPHERE',
                 dims=None, material=None):
        super(Sphere, self).__init__(name, material)
        self.r = r
        self.centers = []
        self.periodic = periodic
        self.tol = tol
        # initialize variables
        self.parts = []
        # update centers
        if center is not None:
            self.add_center(center, dims)

    def _center_exists(self, cmp_center):
        exists = False
        d = len(cmp_center)
        for center in self.centers:
            k = 0
            for elem_center, elem_cmpcenter in zip(center, cmp_center):
                if abs(elem_center - elem_cmpcenter) < self.tol:
                    k += 1
            if k == d:
                exists = True
                break

        return exists

    def _is_inside(self, center, dims):
        dist_squared = self.r**2
        for (c, dim) in zip(center, dims):
            if c < 0.:
                dist_squared -= c**2
            elif c > dim:
                dist_squared -= (c - dim)**2

        return dist_squared > 0

    def add_center(self, center, dims=None):
        if self._center_exists(center) or (dims is not None and not self._is_inside(center, dims)):
            return
        else:
            self.centers.append(center)
            if not self.periodic or dims is None:
                return

        for i, (pos_center, dim) in enumerate(zip(center, dims)):
            if (pos_center + self.r) > dim:
                new_center = copy.copy(center)
                new_center[i] -= dim
                self.add_center(new_center, dims)
            elif (pos_center - self.r) < 0:
                new_center = copy.copy(center)
                new_center[i] += dim
                self.add_center(new_center, dims)

    def create_part(self, model, rve_info=None):

        for i, center in enumerate(self.centers):
            name = '{}_{}'.format(self.name, i)
            self._create_part_by_center(model, center, name, rve_info)

    def _create_part_by_center(self, model, center, name, rve_info,):
        a, b = center[1] + self.r, center[1] - self.r

        # sketch
        sketch = model.ConstrainedSketch(name=name + '_PROFILE',
                                         sheetSize=2 * self.r)
        sketch.ConstructionLine(point1=(center[0], self.r), point2=(center[0], -self.r))
        sketch.ArcByCenterEnds(center=center[:2], point1=(center[0], a),
                               point2=(center[0], b), direction=CLOCKWISE)
        sketch.Line(point1=(center[0], a), point2=(center[0], b))

        # part
        part = model.Part(name=name, dimensionality=THREE_D,
                          type=DEFORMABLE_BODY)
        part.BaseSolidRevolve(sketch=sketch, angle=360.,)
        self.parts.append(part)

        # partitions for meshing
        self._create_partitions(center, part)

        # remove cells
        if rve_info is not None:
            self._remove_cells(center, part, rve_info)

        # assign section
        if self.material is not None:
            self._assign_section(part)

    def _assign_section(self, part):

        # assign section
        part.SectionAssignment(region=(part.cells,),
                               sectionName=self.material.section.name,
                               thicknessAssignment=FROM_SECTION)

    def _create_partitions(self, center, part):
        planes = [YZPLANE, XZPLANE, XYPLANE]
        for c, plane in zip(center, planes):
            offset = c if plane is not XYPLANE else 0.
            feature = part.DatumPlaneByPrincipalPlane(principalPlane=plane,
                                                      offset=offset)
            datum = part.datums[feature.id]
            part.PartitionCellByDatumPlane(datumPlane=datum, cells=part.cells)

    def _remove_cells(self, center, part, rve_info):

        # initialization
        planes = [YZPLANE, XZPLANE, XYPLANE]
        variables = ['x', 'y', 'z']

        # delete cells
        for i in range(3):
            # partition position
            if (center[i] + self.r) > rve_info.dims[i]:
                sign = 1
            elif (center[i] - self.r) < 0.:
                sign = -1
            else:
                continue

            # partition by datum
            if sign > 0:
                x_max = rve_info.dims[i] if i != 2 else rve_info.dims[i] - center[i]
            else:
                x_max = 0. if i != 2 else -center[i]
            feature = part.DatumPlaneByPrincipalPlane(principalPlane=planes[i],
                                                      offset=x_max)
            datum = part.datums[feature.id]
            try:
                part.PartitionCellByDatumPlane(datumPlane=datum, cells=part.cells)
            except:  # in case partition already exists
                pass
            var_name = '{}Max'.format(variables[i]) if sign == -1 else '{}Min'.format(variables[i])
            kwargs = {var_name: x_max}
            faces = part.faces.getByBoundingBox(**kwargs)
            faces_to_delete = []
            for face in faces:
                if abs(face.getNormal()[i]) != 1.0 or (sign == 1 and face.pointOn[0][i] - self.tol > x_max) or (sign == -1 and face.pointOn[0][i] + self.tol < x_max):
                    faces_to_delete.append(face)

            # remove faces
            try:
                part.RemoveFaces(faceList=faces_to_delete, deleteCells=False)
            except:  # in case faces where already removed
                pass

    def create_instance(self, model):

        # create instance
        instances = []
        for i, (center, part) in enumerate(zip(self.centers, self.parts)):
            name = '{}_{}'.format(self.name, i)
            instance = model.rootAssembly.Instance(name=name,
                                                   part=part, dependent=ON)
            instance.translate(vector=(0., 0., center[2]))
            instances.append(instance)

        return instances

    def generate_mesh(self):
        for part in self.parts:
            part.seedPart(size=self.mesh_size,
                          deviationFactor=self.mesh_deviation_factor,
                          minSizeFactor=self.mesh_min_size_factor)

            part.generateMesh()
