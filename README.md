
 - GitHub Setup & Git Commands Guide**  

🔹 Step 1: Create a GitHub Repository**
1. Go to [GitHub](https://github.com/) and log in.
2. Click **New Repository**.
3. Enter a repository name (e.g., `ML_Projects`).
4. Choose **Public** or **Private**.
5. Click **Create Repository**.
6. Copy the repository URL for later use.

---

🔹 Step 2: Set Up Git Locally**
1. Check if Git is Installed**  
   ```bash
   git --version
   ```
   If Git is not installed, install it using:  
   - **Mac:** `brew install git`
   - **Linux:** `sudo apt install git`
   - **Windows:** [Download Git](https://git-scm.com/downloads)

2. Set Global Username & Email (One-time Setup)**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your_email@example.com"
   ```

---

 🔹 Step 3: Connect Local Folder to GitHub**
(A) If You Have an Existing Folder**
```bash
cd path/to/your/project      # Navigate to your project directory
git init                     # Initialize Git in this directory
git remote add origin git@github.com:your-username/your-repo.git  # Add GitHub repo (for SSH)
git remote -v                # Verify the remote repository
```
(B) If You Want to Clone an Existing GitHub Repo**
```bash
git clone git@github.com:your-username/your-repo.git  # Clone the repository (SSH)
cd your-repo
```

---

🔹 Step 4: Add and Push Files to GitHub**
```bash
git add .                      # Add all files to staging
git commit -m "Initial commit"  # Commit with a message
git push origin main            # Push code to GitHub
```

---

🔹 Step 5: Handling SSH Authentication**
1. Check if You Have an SSH Key**
   ```bash
   ls -al ~/.ssh
   ```
   If `id_rsa.pub` exists, you already have a key.

2. Generate a New SSH Key (If Needed)**
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```
   - Press **Enter** to accept the default file location.
   - Leave the passphrase **empty**.

3. Copy the SSH Key and Add It to GitHub**
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
   - Copy the output.
   - Go to **GitHub → Settings → SSH and GPG Keys → New SSH Key**.
   - Paste the key and save.

4. Test SSH Connection**
   ```bash
   ssh -T git@github.com
   ```
   If successful, you’ll see:
   ```
   Hi your-username! You've successfully authenticated.
   ```

---

🔹 Step 6: Common Git Commands**
| Command | Description |
|---------|-------------|
| `git status` | Check modified & untracked files |
| `git add .` | Stage all changes |
| `git commit -m "message"` | Commit changes with a message |
| `git pull origin main --rebase` | Sync local branch with GitHub |
| `git push origin main` | Push changes to GitHub |
| `git log --oneline` | Show commit history |
| `git branch -a` | List all branches |
| `git checkout -b new-branch` | Create and switch to a new branch |
| `git merge branch-name` | Merge branch into main |

---

🔹 Step 7: Updating Existing Files & Pushing Changes**
1. **Check File Changes**
   ```bash
   git status
   ```
2. **Stage Updated Files**
   ```bash
   git add filename_or_folder
   ```
3. **Commit Changes**
   ```bash
   git commit -m "Updated notebook"
   ```
4. **Push Changes to GitHub**
   ```bash
   git push origin main
   ```

---

🔹 Step 8: Fixing Common Errors**
#### **Error: "fatal: origin does not appear to be a git repository"**
Fix:
```bash
git remote set-url origin git@github.com:your-username/your-repo.git
```

#### **Error: "remote: Support for password authentication was removed"**
Fix:
- Use SSH instead of HTTPS.
- Or generate a **Personal Access Token (PAT)** and use it instead of a password.

---

ss🔹 Conclusion**
You have now successfully connected GitHub, initialized Git, and learned essential Git commands to manage your project! 🚀

---
# ML_Projects
