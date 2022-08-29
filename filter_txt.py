import numpy as np

# open('data/ShapeNet55-34/ShapeNet-55/test__bowl.txt','w').writelines([ line for line in open('data/ShapeNet55-34/ShapeNet-55/test_original.txt') if '02880940' in line])
# open('data/ShapeNet55-34/ShapeNet-55/test__mug.txt','w').writelines([ line for line in open('data/ShapeNet55-34/ShapeNet-55/test_original.txt') if '03797390' in line])
# open('data/ShapeNet55-34/ShapeNet-55/test__knife.txt','w').writelines([ line for line in open('data/ShapeNet55-34/ShapeNet-55/test_original.txt') if '03624134' in line])

pcd1 = np.loadtxt('data4.txt')


idx = pcd1[:,-1]==6
bowl_pcd = pcd1[idx]

bowl_pcd = bowl_pcd[:,[0,1,2,-1]]

np.savetxt('data4_bowl.xyz', bowl_pcd[:,:3])
# atoms = []
# coordinates = []

# xyz = open('data3.xyz','r')
# n_atoms = int(xyz.readline())
# title = xyz.readline()
# for line in xyz:
#     x,y,z, id, seg = line.split()
#     coordinates.append([float(x), float(y), float(z), int(seg)])
# xyz.close()

# print(coordinates)