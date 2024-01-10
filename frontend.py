import tkinter as tk
from tkinter import messagebox

class LoginRegisterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Login and Register")

        # Variables to store entered username and password
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        # Create and set up the login frame
        self.login_frame = tk.Frame(root)
        self.login_frame.pack(padx=20, pady=20)
        self.create_login_widgets()

        # Create and set up the register frame
        self.register_frame = tk.Frame(root)
        self.create_register_widgets()

        # Show the login frame by default
        self.show_login_frame()

    def create_login_widgets(self):
        # Login Title
        login_title = tk.Label(self.login_frame, text="Login", font=("Helvetica", 16))
        login_title.grid(row=0, column=0, columnspan=2, pady=10)

        # Username Entry
        username_label = tk.Label(self.login_frame, text="Username:")
        username_label.grid(row=1, column=0, sticky="e")

        username_entry = tk.Entry(self.login_frame, textvariable=self.username_var)
        username_entry.grid(row=1, column=1, pady=5)

        # Password Entry
        password_label = tk.Label(self.login_frame, text="Password:")
        password_label.grid(row=2, column=0, sticky="e")

        password_entry = tk.Entry(self.login_frame, textvariable=self.password_var, show="*")
        password_entry.grid(row=2, column=1, pady=5)

        # Login Button
        login_button = tk.Button(self.login_frame, text="Login", command=self.login)
        login_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Register Button
        register_button = tk.Button(self.login_frame, text="Register", command=self.show_register_frame)
        register_button.grid(row=4, column=0, columnspan=2)

    def create_register_widgets(self):
        # Register Title
        register_title = tk.Label(self.register_frame, text="Register", font=("Helvetica", 16))
        register_title.grid(row=0, column=0, columnspan=2, pady=10)

        # Username Entry
        reg_username_label = tk.Label(self.register_frame, text="Username:")
        reg_username_label.grid(row=1, column=0, sticky="e")

        reg_username_entry = tk.Entry(self.register_frame, textvariable=self.username_var)
        reg_username_entry.grid(row=1, column=1, pady=5)

        # Password Entry
        reg_password_label = tk.Label(self.register_frame, text="Password:")
        reg_password_label.grid(row=2, column=0, sticky="e")

        reg_password_entry = tk.Entry(self.register_frame, textvariable=self.password_var, show="*")
        reg_password_entry.grid(row=2, column=1, pady=5)

        # Register Button
        reg_button = tk.Button(self.register_frame, text="Register", command=self.register)
        reg_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Back to Login Button
        back_button = tk.Button(self.register_frame, text="Back to Login", command=self.show_login_frame)
        back_button.grid(row=4, column=0, columnspan=2)

    def show_login_frame(self):
        self.register_frame.pack_forget()
        self.login_frame.pack()

    def show_register_frame(self):
        self.login_frame.pack_forget()
        self.register_frame.pack()

    def login(self):
        # Dummy login validation, replace this with your actual login logic
        if self.username_var.get() == "user" and self.password_var.get() == "pass":
            messagebox.showinfo("Login Successful", "Welcome, {}".format(self.username_var.get()))
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")

    def register(self):
        # Dummy registration logic, replace this with your actual registration logic
        messagebox.showinfo("Registration Successful", "Account created for {}".format(self.username_var.get()))
        self.show_login_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = LoginRegisterApp(root)
    root.mainloop()
