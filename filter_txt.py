open('data/ShapeNet55-34/ShapeNet-55/test__bowl.txt','w').writelines([ line for line in open('data/ShapeNet55-34/ShapeNet-55/test_original.txt') if '02880940' in line])
open('data/ShapeNet55-34/ShapeNet-55/test__mug.txt','w').writelines([ line for line in open('data/ShapeNet55-34/ShapeNet-55/test_original.txt') if '03797390' in line])
open('data/ShapeNet55-34/ShapeNet-55/test__knife.txt','w').writelines([ line for line in open('data/ShapeNet55-34/ShapeNet-55/test_original.txt') if '03624134' in line])
