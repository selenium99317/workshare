# workshare
sharing work stuffs

to increase performance of games
### gamemoderun RADV_PERFTEST=gpl %command% -USEALLAVAILABLECORES
## OR
### gamemoderun RADV_PERFTEST=gpl PROTON_LOCAL_SHADER_CACHE=1 %command%

### uninstall lutris
sudo pacman -R lutris
sudo pacman -Rsu $(pacman -Qdtq)
