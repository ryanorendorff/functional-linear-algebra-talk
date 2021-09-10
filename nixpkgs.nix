# nixpkgs 21.05 branch from September 1st, 2021
# 87012577f8dfd59e2b68b8c5f1f442db8b40d2a7
import (builtins.fetchTarball {
  url =
    "https://github.com/NixOS/nixpkgs/archive/110a2c9ebbf5d4a94486854f18a37a938cfacbbb.tar.gz";
  sha256 = "0v12ylqxy1kl06dgln6h5k8vhlfzp8xvdymljj7bl0avr0nrgrcm";
}) { }
