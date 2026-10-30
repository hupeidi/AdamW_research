param(
    [string]$PythonExe = 'python',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) { throw "Python not found: $PythonExe" }
    & $PythonExe code/checks.py
    if ($LASTEXITCODE -ne 0) { throw 'Numerical checks failed.' }
    & $PythonExe -u code/train.py --stage pilot --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Pilot experiment failed.' }
    & $PythonExe -u code/train.py --stage main --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Main experiment failed.' }
    & $PythonExe code/analyze.py
    if ($LASTEXITCODE -ne 0) { throw 'Analysis failed.' }
    Write-Host 'Completed. Experimental plots and measurements are in results/.'
} finally {
    Pop-Location
}
