import json
import bpy

with open('./map.json', 'w') as fp:
    json.dump(
        {
            'trees': [
                o.location[0:2]
                    for o in bpy.data.objects
                    if o.name.startswith('Tree')
            ],
            'circles': [
                { 'pos': o.location[0:2], 'radius': o.scale[0] / 2.0 }
                    for o in bpy.data.objects
                    if o.name.startswith('Circle')
            ]
        },
        fp
    )