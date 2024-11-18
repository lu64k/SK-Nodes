this customed nodes pack is some randome works during my own task.

1. sk random file name node: allow user to reading file name randomly or by index from a certain path and output the strings. when the value of "control after generate" is fix, it will read the file name by index.here is the example:
   ![image](https://github.com/user-attachments/assets/d5e611fb-4529-47b2-9967-9222e5da15fc)

3. openai text node is for a post request using openai json format.
4. image tracing node is a image quantize function used for postprocess.this nodes is design for color-picking for who need the generated image as color reference:
   ![image](https://github.com/user-attachments/assets/e48e9d4c-f8aa-48a5-b23b-a892cd083df4)
   ![image](https://github.com/user-attachments/assets/5c9e3654-484c-4ae6-bc9a-1d139003686f)

6. natural saturation: another postprocess node
7. grey scale blend: another postprocess to overlay color info to a grey scale image
   ![image](https://github.com/user-attachments/assets/c26189e9-a3bd-41ed-a670-8ded37a9c6c8)

9. load nemotron: implement of nemotron mini 4b model.
