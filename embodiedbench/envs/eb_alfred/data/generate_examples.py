import os
import json
import embodiedbench.envs.eb_alfred.utils as utils

data_path = os.path.join(os.path.dirname(__file__), 'alfred_prompt_examples.json')
eval_set = 'valid_seen'
example_num = 50

action_space = ['find a Cart', 'find a Potato', 'find a Faucet', 'find a Ottoman', 'find a CoffeeMachine', 'find a Candle', 'find a CD', 'find a Pan', 'find a Watch', 'find a HandTowel', 'find a SprayBottle', 'find a BaseballBat', 'find a CellPhone', 'find a Kettle', 'find a Mug', 'find a StoveBurner', 'find a Bowl', 'find a Toilet', 'find a DiningTable', 'find a Spoon', 'find a TissueBox', 'find a Shelf', 'find a Apple', 'find a TennisRacket', 'find a SoapBar', 'find a Cloth', 'find a Plunger', 'find a FloorLamp', 'find a ToiletPaperHanger', 'find a CoffeeTable', 'find a Spatula', 'find a Plate', 'find a Bed', 'find a Glassbottle', 'find a Knife', 'find a Tomato', 'find a ButterKnife', 'find a Dresser', 'find a Microwave', 'find a CounterTop', 'find a GarbageCan', 'find a WateringCan', 'find a Vase', 'find a ArmChair', 'find a Safe', 'find a KeyChain', 'find a Pot', 'find a Pen', 'find a Cabinet', 'find a Desk', 'find a Newspaper', 'find a Drawer', 'find a Sofa', 'find a Bread', 'find a Book', 'find a Lettuce', 'find a CreditCard', 'find a AlarmClock', 'find a ToiletPaper', 'find a SideTable', 'find a Fork', 'find a Box', 'find a Egg', 'find a DeskLamp', 'find a Ladle', 'find a WineBottle', 'find a Pencil', 'find a Laptop', 'find a RemoteControl', 'find a BasketBall', 'find a DishSponge', 'find a Cup', 'find a SaltShaker', 'find a PepperShaker', 'find a Pillow', 'find a Bathtub', 'find a SoapBottle', 'find a Statue', 'find a Fridge', 'find a Sink', 'pick up the KeyChain', 'pick up the Potato', 'pick up the Pot', 'pick up the Pen', 'pick up the Candle', 'pick up the CD', 'pick up the Pan', 'pick up the Watch', 'pick up the Newspaper', 'pick up the HandTowel', 'pick up the SprayBottle', 'pick up the BaseballBat', 'pick up the Bread', 'pick up the CellPhone', 'pick up the Book', 'pick up the Lettuce', 'pick up the CreditCard', 'pick up the Mug', 'pick up the AlarmClock', 'pick up the Kettle', 'pick up the ToiletPaper', 'pick up the Bowl', 'pick up the Fork', 'pick up the Box', 'pick up the Egg', 'pick up the Spoon', 'pick up the TissueBox', 'pick up the Apple', 'pick up the TennisRacket', 'pick up the Ladle', 'pick up the WineBottle', 'pick up the Cloth', 'pick up the Plunger', 'pick up the SoapBar', 'pick up the Pencil', 'pick up the Laptop', 'pick up the RemoteControl', 'pick up the BasketBall', 'pick up the DishSponge', 'pick up the Cup', 'pick up the Spatula', 'pick up the SaltShaker', 'pick up the Plate', 'pick up the PepperShaker', 'pick up the Pillow', 'pick up the Glassbottle', 'pick up the SoapBottle', 'pick up the Knife', 'pick up the Statue', 'pick up the Tomato', 'pick up the ButterKnife', 'pick up the WateringCan', 'pick up the Vase', 'put down the object in hand', 'drop the object in hand', 'open the Safe', 'close the Safe', 'open the Laptop', 'close the Laptop', 'open the Fridge', 'close the Fridge', 'open the Box', 'close the Box', 'open the Microwave', 'close the Microwave', 'open the Cabinet', 'close the Cabinet', 'open the Drawer', 'close the Drawer', 'turn on the Microwave', 'turn off the Microwave', 'turn on the DeskLamp', 'turn off the DeskLamp', 'turn on the FloorLamp', 'turn off the FloorLamp', 'turn on the Faucet', 'turn off the Faucet', 'slice the Potato', 'slice the Lettuce', 'slice the Tomato', 'slice the Apple', 'slice the Bread']

def load_dataset():
    with open(data_path) as f:
        dataset = json.load(f)
    select_every = len(dataset) // example_num
    dataset = dataset[0:len(dataset):select_every]
    return dataset

example_data = load_dataset()
examples = []
for data in example_data:
    instruction = data['task description']
    plan = data['NL steps']
    real_plan = []
    for action in plan:
        if 'put down' in action:
            real_plan.append('put down the object in hand')
            continue
        if 'drop' in action:
            real_plan.append('drop the object in hand')
            continue

        if ' a ' in action:
            split = ' a '
        elif ' an ' in action:
            split = ' an '
        elif ' the ' in action:
            split = ' the '
        else:
            raise NotImplementedError
        
        action_split = action.split(split)
        act =  action_split[0]
        obj = action_split[1]
        if ' ' in obj:
            obj = ''.join([x.capitalize() for x in obj.split(' ')])
        else:
            obj = obj.capitalize()
        
        if obj == 'Cd':
            obj = 'CD'
        if act in ['pick up', 'turn on', 'turn off', 'slice', 'open', 'close']:
            real_plan.append(act + ' the ' + obj)
        else:
            real_plan.append(act + ' a ' + obj)
    
    action_ids = [action_space.index(x) for x in real_plan]
    action_str = '\n'.join([f'{{"action_id": {x}, "action_name": "{y}"}}' for x, y in zip(action_ids, real_plan)])
    prompt = '''Human instruction: {}.
Output: {{
'language_plan': '',
'executable_plan': [
{}
]}}'''.format(instruction.rstrip('.'), action_str)
    examples.append(prompt)
    # import pdb;pdb.set_trace()

with open('generated_examples_50.json', 'w+') as file:
    json.dump(examples, file, indent=4)
# for res in examples:
#     print(res)

# for res in examples:
#     print('''A robot is working in a house. Below is the instruction, and executable plan for the robot, write the language plan for the robot step by step: {}. 
#     Add your result into the corresponding place in the output dict'''.format(res))


# using gpt4o to 