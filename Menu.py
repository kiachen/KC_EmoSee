from tkinter import *
from PIL import ImageTk, Image
from WebCam import EmoDetection

class Menu(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title('Face Expression Recognition')
        self.iconbitmap('icon/FERicon.ico')
        self.geometry('1000x700')
        self.resizable(width=False,height=False)

        self.label = Label(self,text = "Face Expression Recognition",font=("Fennario",48,"bold")).pack(side="top", fill="x", pady=(10,20))
        self.canvas = Canvas(self, width=750, height=380)
        self.canvas.pack()
        self.original = Image.open("images/menuImg.png")
        resized = self.original.resize((750, 380), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)
        self.canvas.image = self.img

        # self.original = Image.open("images/menuImg.png")
        # resized = self.original.resize((1920, 1314), Image.ANTIALIAS)
        # self.image = ImageTk.PhotoImage(resized)  # Keep a reference, prevent GC
        # self.display = Label(self, image=self.image)




        #define button
        self.btnStart = Button(self, text="Start",font=("Fennario",16,"bold"),width=15,height=3,bd=3,command=self.hideMenu).pack(pady=(50,20))

    def hideMenu(self):
        self.withdraw()
        EmoDetection()
        self.deiconify()

window = Menu()
window.mainloop()