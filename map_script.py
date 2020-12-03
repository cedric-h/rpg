import json
import bpy

with open('./map.json', 'w') as fp:
    o = {
        'trees': [
            o.location[:2]
                for o in bpy.data.objects
                if o.name.startswith('Tree')
        ],
        'brambles': [
            o.location[:2]
                for o in bpy.data.objects
                if o.name.startswith('Bramble')
        ],
        'circles': [
            { 'pos': o.location[:2], 'radius': o.scale[0] / 2.0 }
                for o in bpy.data.objects
                if o.name.startswith('Circle')
        ],
    };
    
    splines = [
        "rocky_enter", "rocky_hide", "rocky_bounce1", "rocky_bounce2",
        "melee_enter", "ranged_enter",
    ]
    
    for spline in splines:
        o[spline] = [
            [ p.co[:2] for p in o.data.splines[0].points ]
                for o in bpy.data.objects
                if o.name == spline
        ][0]
    
    json.dump(o, fp)