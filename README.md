# PROJE

Bu proje yerel olarak `c:\Users\ASUS\Desktop\PROJE` klasöründe bulunuyor.

Hızlı kurulum ve GitHub'a yükleme adımları:

1. Yerel depo oluşturma (zaten yapıldı veya otomatikleştirilebilir):

```
git init
git add .
git commit -m "Initial commit"
```

2. GitHub'da yeni bir repo oluşturun (web arayüzü) veya `gh` CLI kullanın:

```
gh repo create <kullanici>/<repo-ismi> --public
# veya web ile repo oluşturduktan sonra:
git remote add origin https://github.com/<kullanici>/<repo-ismi>.git
git branch -M main
git push -u origin main
```

3. Dikkat: Model ağırlık dosyaları (`*.pt`, `*.pth`) büyük olabilir. GitHub'ın tek dosya sınırı ve depo boyutu göz önünde bulundurulmalıdır.
- Büyük dosyalar için `git lfs` kullanın:

```
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add <model.pt>
git commit -m "Add large model with LFS"
git push
```

Eğer isterseniz, sizin için uzak repo oluşturma adımını (veya `gh` ile otomatik oluşturmayı) ben de çalıştırabilirim — sadece GitHub erişim izni (PAT) veya `gh` CLI kurulu olup olmadığı bilgisini verin.
