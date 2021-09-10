let
  pkgs = import ./nixpkgs.nix;

  # Convenient for using the existing .gitignore to automatically untrack
  # unwanted files in src/.
  inherit (import (pkgs.fetchFromGitHub {
    owner = "hercules-ci";
    repo = "gitignore";
    rev = "211907489e9f198594c0eb0ca9256a1949c9d412";
    sha256 = "06j7wpvj54khw0z10fjyi31kpafkr6hi1k0di13k1xp8kywvfyx8";
  }) { inherit (pkgs) lib; })
    gitignoreSource;

  texlive = pkgs.texlive.combine {
    inherit (pkgs.texlive)
      scheme-small fontspec pgfopts epsdice beamer beamertheme-metropolis;
  };

  fonts = pkgs.makeFontsConf {
    # Currently need Fira because I have not figured out how to replace the sans
    # with the Iosevka sans font.
    fontDirectories = with pkgs; [ fira fira-mono iosevka ];
  };

in pkgs.stdenv.mkDerivation {
  name = "FormalizingLinearAlgebraAlgorithmsPresentation";
  src = gitignoreSource ./src;

  phases = [ "unpackPhase" "buildPhase" ];

  buildInputs = with pkgs; [ texlive pandoc ];

  FONTCONFIG_FILE = "${fonts}";

  buildPhase = ''
    mkdir $out
    pandoc FormalizingLinearAlgebraAlgorithms.lagda.md \
           -o $out/FormalizingLinearAlgebraAlgorithms.pdf \
           -t beamer \
           --slide-level=2 \
           --pdf-engine=xelatex

    pandoc FormalizingLinearAlgebraAlgorithms.lagda.md \
           -o $out/FormalizingLinearAlgebraAlgorithms-handout.pdf \
           -t beamer \
           --slide-level=2 \
           --pdf-engine=xelatex \
           -V handout
  '';
}
